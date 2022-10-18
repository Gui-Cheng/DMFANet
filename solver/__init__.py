#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-17 22:19:38
LastEditTime: 2020-11-12 19:43:57
@Description: file content
'''
# from model.GS import GS
# from model.MTF_GLP import MTF_GLP
# from model.Brovey import Brovey
# # from model.SFIM import SFIM
# from model.CNMF import CNMF

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
from utils.utils import evaluating_config, DIRCNN_data_preprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

np.seterr(divide='ignore', invalid='ignore')


class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)

        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        self.model = net(
            num_channels=self.cfg['data']['n_colors'],
            base_filter=64,
            args=self.cfg
        )

    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])  # 为CPU设置种子用于生成随机数，以使得结果是确定的
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])  # 为当前GPU设置随机种子
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
        psnr, ssim, sam, ergas, scc, q = [], [], [], [], [], []
        for batch in self.data_loader:
            with torch.no_grad():
                ms_image, lms_image, pan_image, bms_image, name = Variable(batch[0]), Variable(
                    batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
            print(ms_image.size())  # [1, 4, 256, 256]
            print(ms_image.squeeze(axis=0).size())  # [4, 256, 256]
            print(ms_image.squeeze().clamp(0, 1).size())  # [4, 256, 256]
            print(ms_image.squeeze().clamp(
                0, 1).numpy().shape)  # (4, 256, 256)

            ms_image = ms_image.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
            lms_image = lms_image.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
            pan_image = pan_image.squeeze(axis=0).clamp(
                0, 1).numpy().transpose(1, 2, 0)
            bms_image = bms_image.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

            # if self.cuda:
            #     ms_image = ms_image.cuda(self.gpu_ids[0])
            #     lms_image = lms_image.cuda(self.gpu_ids[0])
            #     pan_image = pan_image.cuda(self.gpu_ids[0])
            #     bms_image = bms_image.cuda(self.gpu_ids[0])

            t0 = time.time()
            # prediction = SFIM(pan_image, lms_image)
            # prediction = Brovey(pan_image, lms_image)
            # prediction = MTF_GLP(pan_image, lms_image)
            # prediction = GS(pan_image, lms_image)
            prediction = CNMF(pan_image, lms_image)
            t1 = time.time()

            if self.cfg['data']['normalize']:
                ms_image = (ms_image+1) / 2
                lms_image = (lms_image+1) / 2
                pan_image = (pan_image+1) / 2
                bms_image = (bms_image+1) / 2
                prediction = (prediction+1) / 2

            print("===> Processing: %s || Timer: %.4f sec." %
                  (name[0], (t1 - t0)))
            evaluating_config(
                'test', "===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))

            ''' evaluating metrics'''
            test_results = self.evaluating(prediction, ms_image)
            psnr.append(test_results[0])
            ssim.append(test_results[1])
            sam.append(test_results[2])
            ergas.append(test_results[3])
            scc.append(test_results[4])
            q.append(test_results[5])

            bms_image = np.uint8(bms_image*255).astype('uint8')
            ms_image = np.uint8(ms_image*255).astype('uint8')
            lms_image = np.uint8(lms_image*255).astype('uint8')

            avg_time.append(t1 - t0)
            self.save_img(bms_image,
                          name[0][0:-4]+'_bic.tif', mode='RGB')  # CMYK
            self.save_img(ms_image,
                          name[0][0:-4]+'_gt.tif', mode='RGB')
            self.save_img(prediction,
                          name[0][0:-4]+'_fusion.tif', mode='RGB')
            #self.save_img(pan_image.cpu().data, name[0][0:-4]+'_pan.tif', mode='CMYK')
            self.save_img(lms_image,
                          name[0][0:-4]+'_lms.tif', mode='RGB')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))
        avg_psnr = np.array(psnr).mean()
        avg_ssim = np.array(ssim).mean()
        avg_sam = np.array(sam).mean()
        avg_ergas = np.array(ergas).mean()
        ave_scc = np.array(scc).mean()
        avg_q = np.array(q).mean()
        print("===> AVG reduced test: PSNR,     SSIM,   SAM,    ERGAS,  SCC,    Q")
        print('                     ', round(avg_psnr, 4), '', round(avg_ssim, 4), '', round(avg_sam, 4),
              '', round(avg_ergas, 4), '', round(ave_scc, 4), '', round(avg_q, 4))

    def eval(self):
        raise NotImplementedError

    def save_img(self, img, img_name, mode):
        # save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        save_img = img

        # save img
        save_dir = os.path.join('results/', self.cfg['test']['type'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_fn = save_dir + '/' + img_name
        # save_img = np.uint8(save_img*255).astype('uint8')  # shape (132,132,4)
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

    def evaluating(self, fused_image, gt):
        # fused_image = fused_image.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        # fused_image = np.uint8(fused_image*255).astype('uint8')
        # gt = gt.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        gt = np.uint8(gt*255).astype('uint8')

        '''evaluating all methods'''
        ref_results = {}
        ref_results.update(
            {'metrics: ': '  PSNR,     SSIM,   SAM,    ERGAS,  SCC,    Q'})
        temp_ref_results = ref_evaluate(fused_image, gt)
        ref_results.update({'PANNet    ': temp_ref_results})

        ''''print result'''
        print('################## reference comparision #######################')
        for index, i in enumerate(ref_results):
            if index == 0:
                print(i, ref_results[i])
                evaluating_config('test', str(i)+str(ref_results[i]))

            else:
                print(i, [round(j, 4) for j in ref_results[i]])
                evaluating_config('test', str(
                    i)+str([round(j, 4) for j in ref_results[i]]))
        print('################## reference comparision #######################')
        return temp_ref_results
