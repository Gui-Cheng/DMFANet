#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:12:52
LastEditTime: 2020-11-14 12:28:15
@Description: file content
'''
import os
import math
import torch
import cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.vgg import VGG
import torch.nn.functional as F
# from model.deepfuse import MEF_SSIM_Loss
import pytorch_ssim
import pytorch_msssim
from scipy import ndimage


def maek_optimizer(opt_type, cfg, params):
    if opt_type == "ADAM":
        optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'], betas=(
            cfg['schedule']['beta1'], cfg['schedule']['beta2']), eps=cfg['schedule']['epsilon'])
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=cfg['schedule']['lr'], momentum=cfg['schedule']['momentum'])
    elif opt_type == "RMSprop":
        optimizer = torch.optim.RMSprop(
            params, lr=cfg['schedule']['lr'], alpha=cfg['schedule']['alpha'])
    else:
        raise ValueError
    return optimizer


def make_loss(loss_type):
    # loss = {}
    if loss_type == "MSE":
        loss = nn.MSELoss(reduction='sum')
    elif loss_type == "L1":
        loss = nn.L1Loss(reduction='sum')
    elif loss_type == "SSIM":
        loss = pytorch_ssim.SSIM(window_size=11)
    elif loss_type == "MSSSIM":
        loss = pytorch_msssim.MSSSIM()
    else:
        raise ValueError
    return loss


def get_path(subdir):
    return os.path.join(subdir)


def save_config(time, log):
    open_type = 'a' if os.path.exists(
        get_path('./log/' + str(time) + '/records.txt'))else 'w'
    log_file = open(get_path('./log/' + str(time) + '/records.txt'), open_type)
    log_file.write(str(log) + '\n')


def save_net_config(time, log):
    open_type = 'a' if os.path.exists(
        get_path('./log/' + str(time) + '/net.txt'))else 'w'
    log_file = open(get_path('./log/' + str(time) + '/net.txt'), open_type)
    log_file.write(str(log) + '\n')


def draw_curve_and_save(x, y, title, filename, precision):
    if not isinstance(x, np.ndarray):
        x = np.array(x).astype(np.int32)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_title(title)

    max_y = np.ceil(y.max() / precision) * precision
    min_y = np.floor(y.min() / precision) * precision
    major_y_step = (max_y - min_y) / 10
    if major_y_step < 0.1:
        major_y_step = 0.1
    # 设置时间间隔
    ax.yaxis.set_major_locator(MultipleLocator(major_y_step))
    # 设置副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(major_y_step))
    ax.yaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='both')
    # ax.legend()
    if (x.shape[0] >= 2):
        axis_range = [x.min(), x.man(), min_y, max_y]
        ax.axis(axis_range)
    ax.plot(x, y)
    plt.savefig(filename)


def calculate_psnr(img1, img2, pixel_range=255, color_mode='rgb'):
    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])
                ).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])
                ).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2
    # img1 and img2 have range [0, pixel_range]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(pixel_range / math.sqrt(mse))


def ssim(img1, img2, pixel_range=255, color_mode='rgb'):
    C1 = (0.01 * pixel_range)**2
    C2 = (0.03 * pixel_range)**2

    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])
                ).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])
                ).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, pixel_range=255):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, pixel_range)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, pixel_range))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), pixel_range)
    else:
        raise ValueError('Wrong input image dimensions.')


def Highpass(tensor_image):
    kernel = np.ones((5, 5))/25.0

    min_batch = tensor_image.size()[0]
    channels = tensor_image.size()[1]
    out_channel = channels
    kernel = torch.FloatTensor(kernel).expand(
        out_channel, channels, 5, 5).cuda()  # if gpu train
    weight = nn.Parameter(data=kernel, requires_grad=False)
    output = F.conv2d(tensor_image, weight, bias=None,
                      stride=1, padding=(2, 2))
    hp = tensor_image - output
    #print('output size{}'.format(output.size()))
    return hp


def evaluating_config(time, log):
    open_type = 'a' if os.path.exists(
        get_path('./log/' + str(time) + '/evaluating.txt'))else 'w'
    log_file = open(get_path('./log/' + str(time) +
                    '/evaluating.txt'), open_type)
    log_file.write(str(log) + '\n')


def DIRCNN_data_preprocess(up_lr_ms, hr_pan):
    size = up_lr_ms.shape
    bs = size[0]
    bs_hhr_pan_batch = []
    for i in range(bs):
        hhr_pan = []
        gradient_temp = []
        img_4 = up_lr_ms[i, :, :, :]  # (c,h,w)
        img_pan = hr_pan[i, 0, :, :]  # (h,w)

        for j in range(4):
            img = img_4[j, :, :]  # (h,w)
            hhr_pan_1 = (img_pan - torch.mean(img_pan))*torch.std(img) / \
                torch.std(img_pan)+torch.mean(img)  # 高斯匹配后的pan(h,w)
            # print('hhr_pan_1.shape{}'.format(hhr_pan_1.shape))
            hhr_pan.append(hhr_pan_1.unsqueeze(0))  # 扩展维度(1,h,w)
        hhr_pan_4 = torch.cat(hhr_pan, dim=0)  # (4,h,w)
        # print('hhr_pan_4.shape{}'.format(hhr_pan_4.shape))
        bs_hhr_pan_batch.append(hhr_pan_4.unsqueeze(0))  # (1,4,h,w)

    bs_hhr_pan = torch.cat(bs_hhr_pan_batch, dim=0)  # (bs,4,h,w)
    # print(bs_hhr_pan.shape)

    grad_x, grad_y = calculate_gradient(up_lr_ms)  # 计算多光谱的水平和垂直梯度(h,w)
    # print(grad_x.shape,grad_y.shape)
    gradient = torch.cat([grad_x, grad_y], dim=1)  # (bs,8,h,w)
    # print(gradient.shape)

    input1 = bs_hhr_pan-up_lr_ms
    input2 = torch.cat([gradient, input1], dim=1)
    # print(input2.shape)

    return input1, input2, bs_hhr_pan


def calculate_gradient(x):
    # x shape 四维(bs,c,h,w)
    in_channels = x.size()[1]
    out_channels = in_channels

    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).expand(
        out_channels, in_channels, 3, 3)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).expand(
        out_channels, in_channels, 3, 3)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
    grad_x = F.conv2d(x, weight_x, bias=None, stride=1, padding=(1, 1))
    grad_y = F.conv2d(x, weight_y, bias=None, stride=1, padding=(1, 1))
    return grad_x, grad_y


def upsample_interp23(image, ratio):

    image = np.transpose(image, (2, 0, 1))

    b, r, c = image.shape

    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942,
                       0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1]
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23

    first = 1
    for z in range(1, np.int(np.log2(ratio))+1):
        I1LRU = np.zeros((b, 2**z*r, 2**z*c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2] = image
            first = 0
        else:
            I1LRU[:, 0:I1LRU.shape[1]:2, 0:I1LRU.shape[2]:2] = image

        for ii in range(0, b):
            t = I1LRU[ii, :, :]
            for j in range(0, t.shape[0]):
                t[j, :] = ndimage.correlate(t[j, :], BaseCoeff, mode='wrap')
            for k in range(0, t.shape[1]):
                t[:, k] = ndimage.correlate(t[:, k], BaseCoeff, mode='wrap')
            I1LRU[ii, :, :] = t
        image = I1LRU

    re_image = np.transpose(image, (1, 2, 0))
    # re_image=np.transpose(I1LRU, (1, 2, 0))

    return re_image
