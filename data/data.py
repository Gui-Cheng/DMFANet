#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-16 19:22:41
LastEditTime: 2020-11-12 17:04:17
@Description: file content
'''
from os.path import join
from torchvision.transforms import Compose, ToTensor
from .dataset import Data, Data_test, Data_eval
from torchvision import transforms
import torch, h5py, numpy
import torch.utils.data as data

def transform():
    return Compose([
        ToTensor(),
    ])
    #把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    
def get_data(cfg, mode):
    data_dir_ms = join(mode, cfg['source_ms'])
    data_dir_pan = join(mode, cfg['source_pan'])
    cfg = cfg
    return Data(data_dir_ms, data_dir_pan, cfg, transform=transform())
    
def get_test_data(cfg, mode):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    cfg = cfg
    return Data_test(data_dir_ms, data_dir_pan, cfg, transform=transform())

def get_eval_data(cfg, data_dir, upscale_factor):
    raise NotImplementedError
    # data_dir = join(cfg['test']['data_dir'], data_dir)
    # return Data_eval(data_dir, upscale_factor, cfg, transform=transform())