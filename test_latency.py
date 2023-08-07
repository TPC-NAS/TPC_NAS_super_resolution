'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os,sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ModelLoader, global_utils
import  train_image_classification as tic
import torch, time
import numpy as np
from evolution_search import get_new_random_structure_str


def __get_latency__(model, batch_size, resolution, channel, gpu, benchmark_repeat_times, fp16):
    device = torch.device('cuda:{}'.format(gpu))
    torch.backends.cudnn.benchmark = True

    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    if fp16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    the_image = torch.randn(batch_size, channel, resolution, resolution, dtype=dtype,
                            device=device)
    model.eval()
    warmup_T = 1
    with torch.no_grad():
        for i in range(warmup_T):
            the_output = model(the_image)
        start_timer = time.time()
        for repeat_count in range(benchmark_repeat_times):
            the_output = model(the_image)

    end_timer = time.time()
    the_latency = (end_timer - start_timer) / float(benchmark_repeat_times) / batch_size
    return the_latency


def get_robust_latency_mean_std(model, batch_size, resolution, channel, gpu, benchmark_repeat_times=30, fp16=False):
    robust_repeat_times = 10
    latency_list = []
    model = model.cuda(gpu)
    for repeat_count in range(robust_repeat_times):
        try:
            the_latency = __get_latency__(model, batch_size, resolution, channel, gpu, benchmark_repeat_times, fp16)
        except Exception as e:
            print(e)
            the_latency = np.inf

        latency_list.append(the_latency)

    pass  # end for
    latency_list.sort()
    avg_latency = np.mean(latency_list[2:8])
    std_latency = np.std(latency_list[2:8])
    return avg_latency, std_latency

def main(opt, argv):
    global_utils.create_logging()

    batch_size_list = [int(x) for x in opt.batch_size_list.split(',')]
    opt.batch_size = 1
    opt = tic.config_dist_env_and_opt(opt)

    # create model
    model = ModelLoader.get_model(opt, argv)

    print('batch_size, latency_per_image')

    for the_batch_size_per_gpu in batch_size_list:

        the_latency, _ = get_robust_latency_mean_std(model=model, batch_size=the_batch_size_per_gpu,
                                                     resolution=opt.input_image_size, channel=3, gpu=opt.gpu,
                                                     benchmark_repeat_times=opt.repeat_times,
                                                     fp16=opt.fp16)
        print('{},{:4g}'.format(the_batch_size_per_gpu, the_latency))

    if opt.dist_mode == 'auto':
        global_utils.release_gpu(opt.gpu)


def get_model_latency(model, batch_size, resolution, in_channels, gpu, repeat_times, fp16, table):
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    the_image = torch.randn(batch_size, in_channels, resolution, resolution, dtype=dtype,
                            device=device)

    model.eval()
    the_block_latency_list = []
    warmup_T = 1
    with torch.no_grad():
        for i in range(warmup_T):
            the_output = model(the_image)
        for repeat_count in range(repeat_times):
            start_timer = time.time()
            for block in model.block_list:
                the_output = block(the_output)
                end_timer = time.time()
                the_block_latency = (end_timer-start_timer) / batch_size
                the_block_latency_list.append(the_block_latency)
                flops = block.get_FLOPs()
                table[block.__name__+"_{}".format(flops)] = the_latency
                start_timer = time.time()

    with torch.no_grad():
        for i in range(warmup_T):
            the_output = model(the_image)
        for repeat_count in range(repeat_times):
            start_timer = time.time()
            the_output = model(the_image)
            end_timer = time.time()

    print("block latency = ", sum(the_block_latency_list))
    print("total latency = ", (end_timer - start_timer)/batch_size)
    end_timer = time.time()
    the_latency = (end_timer - start_timer) / float(repeat_times) / batch_size
    return the_latency


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=None, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('--repeat_times', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--fp16', action='store_true')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    args = parse_cmd_options(sys.argv)
    table = {}

    select_search_space = global_utils.load_py_module_from_path(args.search_space)
    AnyPlainNet = Masternet.MasterNet

    masternet = AnyPlainNet(num_classes=args.num_classes, opt=args, argv=argv, no_create=True)
    initial_structure_str = str(masternet)

    random_structure_str = get_new_random_structure_str(
                AnyPlainNet=AnyPlainNet, structure_str=initial_structure_str, num_classes=args.num_classes,
                get_search_space_func=select_search_space.gen_search_space, num_replaces=1)

    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=True)
    the_model = the_model.cuda(args.gpu)

    the_latency = get_model_latency(model=the_model, batch_size=args.batch_size,
                                    resolution=args.input_image_size,
                                    in_channels=3, gpu=args.gpu, repeat_times=args.repeat_times,
                                    fp16=args.fp16, table=table)

    print(f'{the_latency:.4g} second(s) per image, or {1.0/the_latency:.4g} image(s) per second.')
