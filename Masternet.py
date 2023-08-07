import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch, argparse
from torch import nn
import torch.nn.functional as F
import PlainNet
from PlainNet import parse_cmd_options, _create_netblock_list_from_str_, basic_blocks, super_blocks
import math
from torch.nn.parameter import Parameter

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_BN', action='store_true')
    parser.add_argument('--no_reslink', action='store_true')
    parser.add_argument('--use_se', action='store_true')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


class MasterNet(PlainNet.PlainNet):
    def __init__(self, argv=None, opt=None, num_classes=None, plainnet_struct=None, no_create=False, scale=2,
                 no_reslink=None, no_BN=None, use_se=None):

        if argv is not None:
            module_opt = parse_cmd_options(argv)
        else:
            module_opt = None

        if no_BN is None:
            if module_opt is not None:
                no_BN = module_opt.no_BN
            else:
                no_BN = False

        if no_reslink is None:
            if module_opt is not None:
                no_reslink = module_opt.no_reslink
            else:
                no_reslink = False

        if use_se is None:
            if module_opt is not None:
                use_se = module_opt.use_se
            else:
                use_se = False


        super().__init__(argv=argv, opt=opt, num_classes=num_classes, plainnet_struct=plainnet_struct, scale=scale,
                                       no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, use_se=use_se)
        # self.last_channels = self.block_list[-1].out_channels
        self.last_channels = self.block_list[0].out_channels

        scaling_factor = scale
        self.iteration = int(math.log2(scaling_factor))

        self.upsample = nn.Sequential(
            *[basic_blocks.SubPixelConvolutionalBlock(kernel_size=3, in_channels=self.last_channels, out_channels=3,  scaling_factor=2, no_create=no_create) for i
              in range(self.iteration)])


        total_channel = 0
        for i in range(len(self.block_list)):
            if i != 0:
                total_channel += self.block_list[i].out_channels

        self.conv     = basic_blocks.ConvKX(in_channels=total_channel, out_channels=self.last_channels, kernel_size=1, stride=1)
        self.SR_conv  = basic_blocks.ConvKX(in_channels=self.last_channels, out_channels=self.last_channels, kernel_size=3, stride=1)

        self.relu = nn.LeakyReLU(0.05, True)
        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se

        # bn eps
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3

    def forward(self, x):
        output = x
        output_buffer_1 = output

        output_buffer = []

        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)

            if block_id == 0 :
                output_buffer_2 = output
            else:
                output_buffer.append(output)

        output    = torch.cat(output_buffer, dim=1)
        output    = self.conv(output)
        output    = self.relu(output)
        output    = self.SR_conv(output) + output_buffer_2

        output    = self.upsample(output)

        return output

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        the_flops += self.conv.get_FLOPs(the_res)
        the_flops += self.SR_conv.get_FLOPs(the_res)
        for i in range(self.iteration):
            the_flops += self.upsample[i].get_FLOPs(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        the_size += self.conv.get_model_size()
        the_size += self.SR_conv.get_model_size()
        for i in range(self.iteration):
            the_size += self.upsample[i].get_model_size()

        return the_size

    def get_num_layers(self):
        num_layers = 0
        for block in self.block_list:
            assert isinstance(block, super_blocks.PlainNetSuperBlockClass)
            num_layers += block.sub_layers
        return num_layers

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block

        if block_id < len(self.block_list) - 1:
            if self.block_list[block_id + 1].in_channels != new_block.out_channels:
                self.block_list[block_id + 1].set_in_channels(new_block.out_channels)
        else:
            assert block_id == len(self.block_list) - 1
            self.last_channels = self.block_list[-1].out_channels
            if self.upsample.in_channels != self.last_channels:
                self.upsample.set_in_channels(self.last_channels)

        self.module_list = nn.ModuleList(self.block_list)

    def split(self, split_layer_threshold):
        new_str = ''
        for block in self.block_list:
            new_str += block.split(split_layer_threshold=split_layer_threshold)
        return new_str

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
