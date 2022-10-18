#!/usr/bin/env python
# coding=utf-8

import torch
import math
import torch.optim as optim
import torch.nn as nn
from importlib import import_module
import torch.nn.functional as F

######################################
#            common model
######################################
class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, activation='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        if scale == 3:
            modules.append(ConvBlock(n_feat, 9 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(3))
            if bn: 
                modules.append(torch.nn.BatchNorm2d(n_feat))
        else:
            for _ in range(int(math.log(scale, 2))):
                modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
                modules.append(torch.nn.PixelShuffle(2))
                if bn: 
                    modules.append(torch.nn.BatchNorm2d(n_feat))
        
        self.up = torch.nn.Sequential(*modules)
        
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        
        if self.pad_model == None:   
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding, bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0, bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class TransConvBlock(ConvBlock):
    def __init__(self, *args, **kwargs):
        super(ConvBlock, self).__init__()
        
        if self.pad_model == None:   
            self.conv = torch.nn.ConvTranspose2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding, bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.ConvTranspose2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0, bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out
                   
######################################
#           resnet_block
###################################### 
 
class ResnetBlock(torch.nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu', norm='batch', pad_model=None):
        super().__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale
        
        if self.norm =='batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        if self.pad_model == None:   
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None, [self.pad, self.conv1, self.normlayer, self.act, self.pad, self.conv2, self.normlayer, self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out
        
class ResnetBlock_triple(ResnetBlock):
    def __init__(self, *args, middle_size, output_size, **kwargs):
        ResnetBlock.__init__(self, *args, **kwargs)

        if self.norm =='batch':
            self.normlayer1 = torch.nn.BatchNorm2d(middle_size)
            self.normlayer2 = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.normlayer1 = torch.nn.InstanceNorm2d(middle_size)
            self.normlayer2 = torch.nn.BatchNorm2d(output_size)
        else:
            self.normlayer1 = None
            self.normlayer2 = None
            
        if self.pad_model == None:   
            self.conv1 = torch.nn.Conv2d(self.input_size, middle_size, self.kernel_size, self.stride, self.padding, bias=self.bias)
            self.conv2 = torch.nn.Conv2d(middle_size, output_size, self.kernel_size, self.stride, self.padding, bias=self.bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad= nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv1 = torch.nn.Conv2d(self.input_size, middle_size, self.kernel_size, self.stride, 0, bias=self.bias)
            self.conv2 = torch.nn.Conv2d(middle_size, output_size, self.kernel_size, self.stride, 0, bias=self.bias)

        layers = filter(lambda x: x is not None, [self.pad, self.conv1, self.normlayer1, self.act, self.pad, self.conv2, self.normlayer2, self.act])
        self.layers = nn.Sequential(*layers) 

    def forward(self, x):

        residual = x
        out = x
        out= self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out

######################################
#         resnet_dense_block
###################################### 

class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return 
        
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out  


class SAM(nn.Module):
    expansion = 1

    def __init__(self):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(12, 12, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU(inplace=True)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.sa(out) * out
        out = self.relu(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class PB(nn.Module):
    def __init__(self):
        super(PB,self).__init__()
        self.sa_1=SAM()
        self.sa_2=SAM()
        self.sa_3=SAM()
        self.sa_4=SAM()
        self.conv=ConvBlock( 48, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
    
    def forward(self, x):
        input=x
        sa1=self.sa_1(x)
        out=sa1+input
        sa2=self.sa_2(out)
        out=sa2+out
        sa3=self.sa_3(out)
        out=sa3+out
        sa4=self.sa_4(out)
        out=torch.cat([sa1,sa2,sa3,sa4],dim=1)
        out=self.conv(out)
        out=input + out
        return out

class PB_1(nn.Module):
    def __init__(self):
        super(PB_1,self).__init__()
        self.sa_1=ConvBlock( 12, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
        self.sa_2=ConvBlock( 12, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
        self.sa_3=ConvBlock( 12, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
        self.sa_4=ConvBlock( 12, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
        self.conv=ConvBlock( 48, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
    
    def forward(self, x):
        input=x
        sa1=self.sa_1(x)
        out=sa1+input
        sa2=self.sa_2(out)
        out=sa2+out
        sa3=self.sa_3(out)
        out=sa3+out
        sa4=self.sa_4(out)
        out=torch.cat([sa1,sa2,sa3,sa4],dim=1)
        out=self.conv(out)
        out=input + out
        return out


class MSB(nn.Module):
    def __init__(self):
        super(MSB,self).__init__()
        self.rcab_1=RCAB()
        self.rcab_2=RCAB()
        self.rcab_3=RCAB()
        self.rcab_4=RCAB()
        self.conv=ConvBlock( 48, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
    
    def forward(self, x):
        input=x
        rcab1=self.rcab_1(x)
        out=rcab1+input
        rcab2=self.rcab_2(out)
        out=rcab2+out
        rcab3=self.rcab_3(out)
        out=rcab3+out
        rcab4=self.rcab_4(out)
        out=torch.cat([rcab1,rcab2,rcab3,rcab4],dim=1)
        out=self.conv(out)
        
        out=input + out
        return out


class MSB_1(nn.Module):
    def __init__(self):
        super(MSB_1,self).__init__()
        self.rcab_1=ConvBlock( 12, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
        self.rcab_2=ConvBlock( 12, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
        self.rcab_3=ConvBlock( 12, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
        self.rcab_4=ConvBlock( 12, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
        self.conv=ConvBlock( 48, 12, 1, 1, 0, activation='relu', norm=None, bias=False)
    
    def forward(self, x):
        input=x
        rcab1=self.rcab_1(x)
        out=rcab1+input
        rcab2=self.rcab_2(out)
        out=rcab2+out
        rcab3=self.rcab_3(out)
        out=rcab3+out
        rcab4=self.rcab_4(out)
        out=torch.cat([rcab1,rcab2,rcab3,rcab4],dim=1)
        out=self.conv(out)
        
        out=input + out
        return out
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat=12, kernel_size=3, reduction=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size,padding=1, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res
