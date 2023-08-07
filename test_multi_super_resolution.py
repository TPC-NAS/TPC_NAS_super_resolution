'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys, copy, time, logging

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

import ModelLoader
import global_utils
from solver import Solver
from model.carn import Net


if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)

    model = ModelLoader.get_model(opt, sys.argv)

    log_filename = os.path.join(opt.save_dir, 'test_image_classification.log')
    global_utils.create_logging(log_filename=log_filename)

    solver = Solver(model, opt)

    solver.load(opt.save_dir, opt.ckpt_name, best="best_Urban100")
    psnr, ssim = solver.evaluate("/home/user2/data/Urban100/", scale=opt.scale, num_step=solver.step)
    print("dataset=Urban100, psnr={}, ssim={}".format(psnr, ssim))

    solver.load(opt.save_dir, opt.ckpt_name, best="best_Set14")
    psnr, ssim = solver.evaluate("/home/user2/data/Set14/", scale=opt.scale, num_step=solver.step)
    print("dataset=Set14   , psnr={}, ssim={}".format(psnr, ssim))

    solver.load(opt.save_dir, opt.ckpt_name, best="best_Set5")
    psnr, ssim = solver.evaluate("/home/user2/data/Set5/", scale=opt.scale, num_step=solver.step)
    print("dataset=Set5    , psnr={}, ssim={}".format(psnr, ssim))

    solver.load(opt.save_dir, opt.ckpt_name, best="best_B100")
    psnr, ssim = solver.evaluate("/home/user2/data/B100/", scale=opt.scale, num_step=solver.step)
    print("dataset=B100    , psnr={}, ssim={}".format(psnr, ssim))
