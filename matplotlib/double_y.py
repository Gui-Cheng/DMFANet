#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 17:46:46
@Description: main.py
'''

import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y1 = [22.1796, 22.8387, 23.1458, 23.4867, 23.9124, 23.6375,
      23.1097, 23.2154, 23.8016, 23.9191, 23.2487, 23.8489]
y2 = [0.7761, 0.8057, 0.8149, 0.8209, 0.8395, 0.8282,
      0.8163, 0.8138, 0.8353, 0.837, 0.8187, 0.8346]

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(x, y1)
ax1.set_ylabel('PSNR')
# ax1.set_title("Double Y axis")

ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, y2, 'r')
ax2.set_ylabel('SSIM')
ax2.set_xlabel('Different level')

plt.show()
