#!/usr/bin/env python
# coding=utf-8


import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # 1
        self.conv_1 = ConvBlock(
            4, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.MSB_1 = MSB()
        self.conv1_pan = ConvBlock(
            1, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.PB_1 = PB()

        # 2
        self.conv_2 = ConvBlock(
            12, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.MSB_2 = MSB()
        self.conv2_pan = ConvBlock(
            12, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.PB_2 = PB()

        # 3
        self.conv_3 = ConvBlock(
            12, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.MSB_3 = MSB()
        self.conv3_pan = ConvBlock(
            12, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.PB_3 = PB()

        # 4
        self.conv_4 = ConvBlock(
            12, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.MSB_4 = MSB()
        self.conv4_pan = ConvBlock(
            12, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.PB_4 = PB()

        # 5
        self.conv_5 = ConvBlock(
            12, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.MSB_5 = MSB()
        self.conv5_pan = ConvBlock(
            12, 12, 3, 1, 1, activation='relu', norm=None, bias=False)
        self.PB_5 = PB()

        self.conv_end = ConvBlock(
            12, 12, 3, 1, 1, activation='relu', norm=None, bias=False)

        self.conv_final = ConvBlock(
            60, 4, 1, 1, 0, activation='relu', norm=None, bias=False)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, l_ms,b_ms, x_pan):

        input_x4 = F.interpolate(l_ms, scale_factor=4,
                                 mode='bicubic', align_corners=True)
        ### layer 1 ####
        layer1_conv = self.conv_1(input_x4)
        
        layer1_ms = self.MSB_1(layer1_conv)
        layer1_pan_conv = self.conv1_pan(x_pan)
        layer1_pan = self.PB_1(layer1_pan_conv)
        
        out1 = layer1_ms+layer1_pan
      

        #### layer 2 ####
        layer2_conv = self.conv_2(out1)
        layer2_ms = self.MSB_2(layer2_conv)
        layer2_pan_conv = self.conv2_pan(layer1_pan)
        layer2_pan = self.PB_2(layer2_pan_conv)
        out2 = layer2_ms + layer2_pan

        #### layer 3 ####
        layer3_conv = self.conv_3(out2)
        layer3_ms = self.MSB_3(layer3_conv)
        layer3_pan_conv = self.conv3_pan(layer2_pan)
        layer3_pan = self.PB_3(layer3_pan_conv)
        out3 = layer3_ms + layer3_pan

        #### layer 4 ####
        layer4_conv = self.conv_4(out3)
        layer4_ms = self.MSB_4(layer4_conv)
        layer4_pan_conv = self.conv4_pan(layer3_pan)
        layer4_pan = self.PB_4(layer4_pan_conv)
        out4 = layer4_ms + layer4_pan

        #### layer 5 ####
        layer5_conv = self.conv_5(out4)
        layer5_ms = self.MSB_5(layer5_conv)
        layer5_pan_conv = self.conv5_pan(layer4_pan)
        layer5_pan = self.PB_5(layer5_pan_conv)
        out5 = layer5_ms + layer5_pan

        #### layer end ####
        layer_last = self.conv_end(out5)

        layer_cat = torch.cat(
            [layer2_conv, layer3_conv, layer4_conv, layer5_conv, layer_last], 1)

        out = self.conv_final(layer_cat)

        return out
