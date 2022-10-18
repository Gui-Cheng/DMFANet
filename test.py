#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 17:50:08
@Description: test.py
'''

from utils.config import get_config
#from solver.testsolver import Testsolver
from solver.testsolver_full_resolution import Testsolver  # full_resolution evaluation
# from solver.testsolver_reduced import Testsolver  # reduced_resolution evaluation
# reduced_resolution traditianal_evaluation
# from solver.testsolver_reduced_traditianal import Testsolver
# full_resolution traditianal_evaluation
# from solver.testsolver_full_resolution_traditional import Testsolver

if __name__ == '__main__':
    cfg = get_config('option.yml')
    solver = Testsolver(cfg)
    solver.run()
