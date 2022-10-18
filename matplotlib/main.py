#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 17:46:46
@Description: main.py
'''

from utils.config import get_config
from solver.solver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N_SR')  #创建解析器，使用 argparse 的第一步是创建一个 ArgumentParser 对象。
    parser.add_argument('--option_path', type=str, default='option.yml')#添加参数，给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的
    opt = parser.parse_args()  #解析参数
    cfg = get_config(opt.option_path)  #参数字典
    solver = Solver(cfg)
    solver.run()
    