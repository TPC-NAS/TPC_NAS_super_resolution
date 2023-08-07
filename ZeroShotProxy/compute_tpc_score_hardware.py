'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''



import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import global_utils, argparse, ModelLoader, time
from PlainNet import basic_blocks
import math
import Masternet
from PlainNet import basic_blocks, super_blocks, SuperResKXKX, SuperResK1KXK1

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def compute_nas_score(s):
    info = {}
    nas_score_list = []

    test_log_conv_scaling_factor = 0
    block_count = 0
    final_out_channel = 0
    total_out_channel = 0

    while len(s) > 0:
        block_count += 1
        # print(input_resolution)
        # print(s)
        tmp_idx_1 = s.find('(')
        tmp_idx_2 = s.find(')')
        block_type      = s[0:tmp_idx_1]
        params_block    = s[tmp_idx_1+1:tmp_idx_2].split(",")
        s = s[tmp_idx_2+1:]

        if len(params_block) == 4 :
            in_channel    = int(params_block[0])
            out_channel   = int(params_block[1])
            stride        = int(params_block[2])
            sublayers     = int(params_block[3])
            final_out_channel = out_channel
        else:
            in_channel    = int(params_block[0])
            out_channel   = int(params_block[1])
            stride        = int(params_block[2])
            bottleneck    = int(params_block[3])
            sublayers     = int(params_block[4])
            total_out_channel += out_channel

        if block_type == "SuperConvK3BNRELU":
            kernel_size = 3
            score = out_channel * (kernel_size**2) / 1.0
        elif block_type == "SuperConvK1BNRELU":
            kernel_size = 1
            score = out_channel * (kernel_size**2) / 1.0
        elif block_type == "SuperResK1K3K1" or block_type == "SuperResK1K5K1" or block_type == "SuperResK1K7K1":
            if block_type == "SuperResK1K3K1":
                kernel_size = 3
            elif block_type == "SuperResK1K5K1":
                kernel_size = 5
            else:
                kernel_size = 7

            score = (((bottleneck**4) * (out_channel**2) * (kernel_size**4))**sublayers) / (stride**2)
            score *= (out_channel//2) ** (sublayers+1) * out_channel * 9

        elif block_type == "SuperResK3K3" or block_type == "SuperResK5K5" or block_type == "SuperResK7K7":
            if block_type == "SuperResK3K3":
                kernel_size = 3
            elif block_type == "SuperResK5K5":
                kernel_size = 5
            else:
                kernel_size = 7

            score = ((bottleneck*out_channel*(kernel_size**4))**sublayers) / (stride**2)
            score *= (out_channel//2) ** (sublayers+1) * out_channel * 9
            # score *= out_channel * (out_channel//2) * 9

        test_log_conv_scaling_factor += math.log(score)

    score =  (final_out_channel ** 2) * (3**4) * 4
    # score = out_channel ** 2 * 3 * (3**4)
    test_log_conv_scaling_factor += math.log(score)

    nas_score = test_log_conv_scaling_factor
    nas_score_list.append(float(nas_score))

    assert not (nas_score != nas_score)

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)

    return info


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)
    the_model = ModelLoader.get_model(opt, sys.argv)
    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)

    from evolution_search_hardware import model_encoder
    AnyPlainNet = Masternet.MasterNet
    structure_str = "SuperConvK3BNRELU(3,64,1,1)SuperResK3K3(64,64,1,64,3)SuperResK3K3(128,128,2,128,4)SuperResK3K3(256,256,2,256,6)SuperResK3K3(512,512,2,512,3)"
    structure_str = model_encoder(AnyPlainNet, structure_str, 1000)

    start_timer = time.time()
    info = compute_nas_score(structure_str)
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info['avg_nas_score']
    print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')
