import os
import random
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
import logging
import math
from PlainNet.basic_blocks import ResBlock, ResBlockProj
import time
import cv2

class Solver():
    def __init__(self, model, cfg):
        self.refiner = model

        if cfg.loss_fn in ["MSE"]:
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]:
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()),
            cfg.lr)

        self.train_data   = TrainDataset(cfg.train_data_path,
                                       scale=cfg.scale,
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=4,
                                       shuffle=True, drop_last=True)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner = self.refiner.to(self.device)
        self.loss_fn = self.loss_fn

        self.cfg = cfg
        self.step = 0
        # self.max_score = 0
        self.psnr_1 = 0
        self.psnr_2 = 0
        self.psnr_3 = 0
        self.psnr_4 = 0

        # self.writer = SummaryWriter(log_dir=os.path.join("runs", cfg.ckpt_name))

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def fit(self):
        cfg = self.cfg
        refiner = self.refiner

        learning_rate = cfg.lr
        while True:
            # print(len(self.train_loader))
            for inputs in self.train_loader:
                self.refiner.train()

                if cfg.scale > 0:
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1]
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary
                    scale = random.randint(2, 4)
                    hr, lr = inputs[scale-2][0], inputs[scale-2][1]

                hr = hr.to(self.device)
                lr = lr.to(self.device)

                # sr = refiner(lr, scale)
                self.optim.zero_grad()
                sr = refiner(lr)
                loss = self.loss_fn(sr, hr)

                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate

                self.step += 1
                if cfg.verbose and self.step % cfg.print_interval == 0:
                    psnr_1, ssim_1 = self.evaluate("/home/user2/data/Urban100/", scale=cfg.scale, num_step=self.step, calculate_ssim=False)
                    psnr_2, ssim_2 = self.evaluate("/home/user2/data/Set14/", scale=cfg.scale, num_step=self.step, calculate_ssim=False)
                    psnr_3, ssim_3 = self.evaluate("/home/user2/data/Set5/", scale=cfg.scale, num_step=self.step, calculate_ssim=False)
                    psnr_4, ssim_4 = self.evaluate("/home/user2/data/B100/", scale=cfg.scale, num_step=self.step, calculate_ssim=False)

                    if psnr_1 > self.psnr_1:
                        self.psnr_1 = psnr_1
                        self.save(cfg.save_dir, cfg.ckpt_name, best="best_Urban100")

                    if psnr_2 > self.psnr_2:
                        self.psnr_2 = psnr_2
                        self.save(cfg.save_dir, cfg.ckpt_name, best="best_Set14")

                    if psnr_3 > self.psnr_3:
                        self.psnr_3 = psnr_3
                        self.save(cfg.save_dir, cfg.ckpt_name, best="best_Set5")

                    if psnr_4 > self.psnr_4:
                        self.psnr_4 = psnr_4
                        self.save(cfg.save_dir, cfg.ckpt_name, best="best_B100")

                    logging.info("step={}/{}, psnr={}, ssim={}".format(self.step, cfg.max_steps, psnr_1, ssim_1))

                    self.save(cfg.save_dir, cfg.ckpt_name)

            if self.step > cfg.max_steps: break

    def evaluate(self, test_data_dir, scale=2, num_step=0, calculate_ssim=True):
        cfg = self.cfg
        mean_psnr = 0
        mean_ssim = 0
        self.refiner.eval()

        test_data   = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

        for step, inputs in enumerate(test_loader):
            hr = inputs[0]
            lr = inputs[1]
            name = inputs[2][0]

            lr_patch = lr
            lr_patch = lr_patch.to(self.device)

            # run refine process in here!
            sr = self.refiner(lr_patch).data

            hr = hr.squeeze(0)
            sr = sr.squeeze(0)

            hr = hr.cpu().mul(255).clamp(0, 255).div(255).permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).div(255).permute(1, 2, 0).numpy()

            im1 = hr;
            im2 = sr;

            psnr_test, ssim_test = psnr(im1, im2, calculate_ssim)
            mean_psnr += psnr_test / len(test_data)
            mean_ssim += ssim_test / len(test_data)

        return mean_psnr, mean_ssim

    def load(self, ckpt_dir, ckpt_name, best=None):
        if best is not None:
            load_path = os.path.join(ckpt_dir, "{}_{}.pth".format(ckpt_name, best))
        else:
            load_path = os.path.join(ckpt_dir, "{}_latest.pth".format(ckpt_name))

        try:
            checkpoint = torch.load(load_path)
        except:
            return
        self.refiner.load_state_dict(checkpoint["model_state_dict"])
        self.step = checkpoint["step"]

        print("Load pretrained {} model".format(load_path))

    def save(self, ckpt_dir, ckpt_name, best=None):
        if best is not None:
            save_path = os.path.join(ckpt_dir, "{}_{}.pth".format(ckpt_name, best))
            # torch.save(self.refiner.state_dict(), save_path)
            torch.save({
                "step": self.step,
                "model_state_dict": self.refiner.state_dict()
                }, save_path)
        else:
            save_path = os.path.join(ckpt_dir, "{}_latest.pth".format(ckpt_name))
            # torch.save(self.refiner.state_dict(), save_path)
            torch.save({
                "step": self.step,
                "model_state_dict": self.refiner.state_dict()
                }, save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr


def psnr(im1, im2, calculate_ssim_or_not):
    im1 = bgr2ycbcr(im1)
    im2 = bgr2ycbcr(im2)

    crop_border = 4

    if im1.ndim == 3:
        im1 = im1[crop_border:-crop_border, crop_border:-crop_border, :]
        im2 = im2[crop_border:-crop_border, crop_border:-crop_border, :]
    elif im1.ndim == 2:
        im1 = im1[crop_border:-crop_border, crop_border:-crop_border]
        im2 = im2[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1.ndim))

    psnr = calculate_psnr(im1 * 255, im2 * 255)

    ssim = 0
    if calculate_ssim_or_not:
        ssim = calculate_ssim(im1 * 255, im2 * 255)

    return psnr, ssim

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    # print(in_img_type)
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
        # rlt = np.dot(img, [65.738, 129.057, 25.064]) / 256.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
