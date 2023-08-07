# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:29:52 2017
@author: LM
"""

import argparse, os,re, sys
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from SRResNet import SRResNet,SRResNet_RGBY
# from data import DIV2K
from torchvision import models
import torch.utils.model_zoo as model_zoo
# from tool import Normalize
# from tool import deNormalize
import ModelLoader
import global_utils
from dataset import TrainDataset, TestDataset
import numpy as np
import scipy.misc as misc
import skimage.measure as measure

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--gpu_num", default="1", type=str, help="which gpu(0 or 1) to use for train")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")

parser.add_argument("--train_data_path", type=str, default="/home/user1/data/DIV2K/DIV2K_train.h5")
parser.add_argument("--ckpt_dir", type=str,default="checkpoint")
parser.add_argument("--shave", type=int, default=20)
parser.add_argument("--patch_size", type=int, default=64)
parser.add_argument('--arch', default=None, help='model names/module to load')
parser.add_argument("--plainnet_struct_txt")
parser.add_argument("--scale", type=int)
parser.add_argument("--no_BN")
parser.add_argument("--num_classes", default=10)
parser.add_argument("--save_dir")
# normal = Normalize(mean = [0.485, 0.456, 0.406],
#                    std = [0.229, 0.224, 0.225])
# deNormal = deNormalize(mean = [0.485, 0.456, 0.406],
#                        std = [0.229, 0.224, 0.225])


def main():

    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===> Loading datasets")
    #dataPath = r'/home/we/devsda1/lm/DIV2K/DIV2K_Ch[RGB][RGB]_Size[24][96]_num[200].npz'
    #train_set = DIV2K(dataPath,in_channels =3,out_channels = 3)
    #dataPath = r'/home/we/devsda1/lm/DIV2K/DIV2K_Ch[Y][Y]_Size[24][96]_num[200].npz'
    #train_set = DIV2K(dataPath,in_channels =1,out_channels = 1)

    # dataPath = r'/home/we/devsda1/lm/DIV2K/DIV2K_Ch[RGB][RGB]_Size[24][96]_num[200].npz'
    # train_set = DIV2K(dataPath,in_channels =3,out_channels = 3)
    # training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    train_set            = TrainDataset(opt.train_data_path,
                                       scale=opt.scale,
                                       size=opt.patch_size)
    training_data_loader = DataLoader(dataset=train_set,
                                       batch_size=opt.batchSize,
                                       num_workers=opt.threads,
                                       shuffle=True, drop_last=True)

    opt.vgg_loss = False
    if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])

            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()

    print("===> Building model")
    # model = SRResNet(in_channels =3,out_channels = 3,bn = False)
    #model=  SRResNet_RGBY(in_channels = 4,out1_channels = 3,out2_channels = 1,bn = False)
    # opt     = global_utils.parse_cmd_options(sys.argv)
    model   = ModelLoader.get_model(opt, sys.argv)
    # model   = init_model(model, opt, sys.argv)

    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.to(device)
        criterion = criterion.to(device)
        if opt.vgg_loss:
            netContent = netContent.to(device)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            # model_epoch_20.pth => 20
            # opt.start_epoch = int(re.split('_',re.split('\.',opt.resume)[-2])[-1]) + 1
            model.load_state_dict(torch.load(opt.resume))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights)
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    with open(os.path.join(opt.save_dir,'train_LR[RGB]HR[RGB].log'),'w') as f:
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            # train(training_data_loader, optimizer, criterion, epoch, f, device)
            psnr = evaluate("/home/user1/data/Urban100/", device, scale=opt.scale)
            print("===> Epoch[{}]: PSNR: {:.10f}".format(epoch, psnr))
            # f.write("===> Epoch[{}]: PSNR: {:.10f}".format(epoch, psnr))
            # save_checkpoint(model, epoch, opt.save_dir)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, criterion, epoch, file, device):
    # adjust learnnig rate,every step reduce to 0.1*lr
    lrate = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lrate

    print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(training_data_loader):
        #for j in range(batch[0].shape[0]):  # batchsize
        #    batch[0] = normal(batch[0])
        #    batch[1] = normal(batch[1])
        # lr, hr = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        hr, lr = batch[-1][0], batch[-1][1]
        # print(hr.shape)
        # print(lr.shape)
        if opt.cuda:
            lr = lr.to(device)
            hr = hr.to(device)
        sr = model(lr)

        mseloss = criterion(sr, hr)

        if opt.vgg_loss:
            content_input = netContent(sr)
            content_target = netContent(hr)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)
            #netContent.zero_grad()
            loss = mseloss + 0.006*content_loss
            #loss = content_loss
        else:
            loss = mseloss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if opt.vgg_loss:
            write_str = '%f\t%f\n' % (loss.data,content_loss.data)
        else:
            write_str = '%f\n' % (loss.data)
        file.write(write_str)
        if iteration%100 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f} Content_loss {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data, content_loss.data))
            else:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data))

def evaluate(test_data_dir, device, scale=2):
    def psnr(im1, im2):
        def im2double(im):
            min_val, max_val = 0, 255
            out = (im.astype(np.float64)-min_val) / (max_val-min_val)
            return out

        im1 = im2double(im1)
        im2 = im2double(im2)
        psnr = measure.compare_psnr(im1, im2, data_range=1)
        return psnr

    cfg = opt
    mean_psnr = 0
    model.eval()

    test_data   = TestDataset(test_data_dir, scale=scale)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             num_workers=1,
                             shuffle=False)

    for step, inputs in enumerate(test_loader):
        hr = inputs[0].squeeze(0)
        lr = inputs[1].squeeze(0)
        name = inputs[2][0]

        h, w = lr.size()[1:]
        h_half, w_half = int(h/2), int(w/2)
        h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

        # split large image to 4 patch to avoid OOM error
        lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
        lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
        lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
        lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
        lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
        lr_patch = lr_patch.to(device)

        # run refine process in here!
        # sr = self.refiner(lr_patch, scale).data
        with torch.no_grad():
            sr = model(lr_patch).data

        h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
        w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale

        # merge splited patch images
        result = torch.FloatTensor(3, h, w).to(device)
        result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
        result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
        result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
        result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
        sr = result

        hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        # evaluate PSNR
        # this evaluation is different to MATLAB version
        # we evaluate PSNR in RGB channel not Y in YCbCR
        bnd = scale
        im1 = hr[bnd:-bnd, bnd:-bnd]
        im2 = sr[bnd:-bnd, bnd:-bnd]
        mean_psnr += psnr(im1, im2) / len(test_data)

    return mean_psnr

def save_checkpoint(model, epoch, save_dir):
    model_out_path = save_dir + "/model_ch[RGB][RGB]_epoch_latest.pth"
    #state = {"epoch": epoch ,"model": model}
    if not os.path.exists("../model/"):
        os.makedirs("../model/")

    #torch.save(state, model_out_path)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
