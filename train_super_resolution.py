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

def network_weight_MSRAPrelu_init(net: nn.Module):
    # the gain of xavier_normal_ is computed from gain=magnitude * sqrt(3) where magnitude is 2/(1+0.25**2). [mxnet implementation]

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data, gain=3.26033)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    return net


def network_weight_xavier_init(net: nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    return net

def network_weight_stupid_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(k1 * k2 * in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def network_weight_zero_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(k1 * k2 * in_channels) * 1e-3
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(in_channels) * 1e-3
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def network_weight_01_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(k1 * k2 * in_channels) * 0.1
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(m.weight.shape, device=device) / np.sqrt(in_channels) * 0.1
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def init_model(model, opt, argv):
    if hasattr(opt, 'weight_init') and opt.weight_init == 'xavier':
        network_weight_xavier_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'MSRAPrelu':
        network_weight_MSRAPrelu_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'stupid':
        network_weight_stupid_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'zero':
        network_weight_zero_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == '01':
        network_weight_01_init(model)
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'custom':
        assert hasattr(model, 'init_parameters')
        model.init_parameters()
    elif hasattr(opt, 'weight_init') and opt.weight_init == 'None':
        logging.info('Warning!!! model loaded without initialization !')
    else:
        raise ValueError('Unknown weight_init')

    if hasattr(opt, 'bn_momentum') and opt.bn_momentum is not None:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = opt.bn_momentum

    if hasattr(opt, 'bn_eps') and opt.bn_eps is not None:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = opt.bn_eps

    return model

if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)

    job_done_fn = os.path.join(opt.save_dir, 'train_super_resolution.done')
    if os.path.isfile(job_done_fn):
        print('skip ' + job_done_fn)
    else:
        model = ModelLoader.get_model(opt, sys.argv)
        model = init_model(model, opt, sys.argv)

        log_filename = os.path.join(opt.save_dir, 'train_super_resolution.log')
        global_utils.create_logging(log_filename=log_filename)

        solver = Solver(model, opt)
        if opt.resume:
            # solver.load(opt.save_dir, opt.ckpt_name)
            solver.load(opt.save_dir, opt.ckpt_name, best="best_Urban100")

        if opt.auto_resume:
            solver.load(opt.save_dir, "ckpt")

        # solver.load(opt.save_dir, opt.ckpt_name)
        solver.fit()

        training_status_info = "finish"
        global_utils.save_pyobj(job_done_fn, training_status_info)

