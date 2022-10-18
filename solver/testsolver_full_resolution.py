#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-17 22:19:38
LastEditTime: 2020-11-12 19:43:57
@Description: file content
'''

from metrics import ref_evaluate, no_ref_evaluate
from solver.basesolver import BaseSolver
import os
import torch
import time
import cv2
import importlib
import torch.backends.cudnn as cudnn
from data.data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch.nn.functional as F
from utils.utils import evaluating_config, DIRCNN_data_preprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)

        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        self.model = net( 
            args=self.cfg
        )

    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])  
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed']) 
            cudnn.benchmark = True

            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >= 0:
                    self.gpu_ids.append(gid)
            torch.cuda.set_device(self.gpu_ids[0])

            self.model_path = os.path.join(
                self.cfg['checkpoint'], self.cfg['test']['model'])

            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.gpu_ids)
            self.model.load_state_dict(torch.load(
                self.model_path, map_location=lambda storage, loc: storage)['net'])

    def test(self):
        self.model.eval()
        avg_time = []
        D_lamda, D_s, QNR = [], [], []
        for batch in self.data_loader:
            with torch.no_grad():
                ms_image, lms_image, pan_image, bms_image, name = Variable(batch[0]), Variable(
                    batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])

      
            if self.cuda:
                ms_image = ms_image.cuda(self.gpu_ids[0])
                # lms_image = lms_image.cuda(self.gpu_ids[0])
                pan_image = pan_image.cuda(self.gpu_ids[0])
                bms_image = bms_image.cuda(self.gpu_ids[0])

            t0 = time.time()
            prediction = self.model( ms_image,bms_image,pan_image)
            t1 = time.time()

            if self.cfg['data']['normalize']:
                ms_image = (ms_image+1) / 2
                # lms_image = (lms_image+1) / 2
                pan_image = (pan_image+1) / 2
                bms_image = (bms_image+1) / 2
                prediction = (prediction+1) / 2

            print("===> Processing: %s || Timer: %.4f sec." %
                  (name[0], (t1 - t0)))
            evaluating_config(
                'test', "===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))

            ''' evaluating metrics'''
            test_results = self.evaluating(prediction.cpu().data, pan_image.cpu().data, ms_image.cpu(
            ).data)

            D_lamda.append(test_results[0])
            D_s.append(test_results[1])
            QNR.append(test_results[2])

            avg_time.append(t1 - t0)
            # self.save_img(bms_image.cpu().data,
            #               name[0][0:-4]+'_bic.tif', mode='RGB')  # CMYK
            self.save_img(ms_image.cpu().data,
                          name[0][0:-4]+'_gt.tif', mode='RGB')
            self.save_img(prediction.cpu().data,
                          name[0][0:-4]+'_fusion.tif', mode='RGB')
            #self.save_img(pan_image.cpu().data, name[0][0:-4]+'_pan.tif', mode='CMYK')
            # self.save_img(lms_image.cpu().data,
            #               name[0][0:-4]+'_lms.tif', mode='RGB')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))
        avg_D_lamda = np.array(D_lamda).mean()
        avg_D_s = np.array(D_s).mean()
        avg_QNR = np.array(QNR).mean()
        print("===> AVG reduced test: D_lamda,     D_s,   QNR")
        print('                     ', round(avg_D_lamda, 4),
              '', round(avg_D_s, 4), '', round(avg_QNR, 4))

    def eval(self):
        raise NotImplementedError

    def save_img(self, img, img_name, mode):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        # save img
        save_dir = os.path.join('results/', self.cfg['test']['type'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_fn = save_dir + '/' + img_name
        save_img = np.uint8(save_img*255).astype('uint8')  # shape (132,132,4)
        save_img = np.stack(
            [save_img[:, :, 2], save_img[:, :, 1], save_img[:, :, 0]], axis=2)  # if save RGB
        save_img = Image.fromarray(save_img, mode)
        save_img.save(save_fn)

    def run(self):
        self.check()
        if self.cfg['test']['type'] == 'test':
            self.dataset = get_test_data(
                self.cfg, self.cfg['test']['data_dir'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                                          num_workers=self.cfg['threads'])
            self.test()
        elif self.cfg['test']['type'] == 'eval':
            raise NotImplementedError
        else:
            raise ValueError('Mode error!')

    def evaluating(self, fused_image, used_pan, used_ms):
        fused_image = fused_image.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        fused_image = np.uint8(fused_image*255).astype('uint8')
        used_pan = used_pan.squeeze().clamp(0, 1).numpy()
        used_pan = np.uint8(used_pan*255).astype('uint8')
        used_pan = np.expand_dims(used_pan, -1)
        used_ms = used_ms.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        used_ms = np.uint8(used_ms*255).astype('uint8')

        '''evaluating all methods'''
        no_ref_results = {}
        no_ref_results.update({'metrics: ': '  D_lamda, D_s,    QNR'})
        temp_no_ref_results = no_ref_evaluate(fused_image, used_pan, used_ms)
        no_ref_results.update({'PANNet    ': temp_no_ref_results})

        ''''print result'''

        print('################## no reference comparision ####################')
        for index, i in enumerate(no_ref_results):
            if index == 0:
                print(i, no_ref_results[i])
                evaluating_config('test', str(i)+str(no_ref_results[i]))
            else:
                print(i, [round(j, 4) for j in no_ref_results[i]])
                evaluating_config('test', str(
                    i)+str([round(j, 4) for j in no_ref_results[i]]))
        print('################## no reference comparision ####################')
        return temp_no_ref_results
