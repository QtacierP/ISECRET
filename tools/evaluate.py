# -*- coding: utf-8 -*-
import sys
import cv2
import glob
import os
from matplotlib.pyplot import imread
from tqdm import tqdm
import numpy as np
import math
from multiprocessing import Pool
import matplotlib.pyplot as plt
import argparse


args = argparse.ArgumentParser(description='the option of the evaluation')

args.add_argument('--test_dir', type=str, default='', help='enhancement dir')
args.add_argument('--gt_dir', type=str, default='', help='gt dir')
args.add_argument('--mask_dir', type=str, default='', help='mask dir')


args = args.parse_args()



def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2):
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
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


gt_list = sorted(glob.glob(os.path.join(args.gt_dir, '*')))
image_list = sorted(glob.glob(os.path.join(args.test_dir, '*')))


def run(source_path, target_path, mask_path=None):
    source = imread(source_path).copy()
    if mask_path is not None:
        mask = imread(mask_path)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    target = imread(target_path).copy()
    source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_NEAREST)
    if mask_path is  not  None:
        source[mask == 0] = 0  
        target[mask == 0] = 0
    target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_NEAREST)
    return psnr(source, target), ssim(source, target)


psnr_list = []
ssim_list = []
pool = Pool(processes=32)
result = []
for image_path in tqdm(image_list):
    try:
        _, image_id = os.path.split(image_path)
        
        if args.mask_dir != '':
            mask_path = os.path.join(args.mask_dir, image_id.split('.')[0] + '_mask.gif')
        else:
            mask_path = None
        image_id_split = image_id.split('_')[:-1]
        image_name = ''
        for idx, word in enumerate(image_id_split):
            image_name += str(word)
            if idx != len(image_id_split) - 1:
                image_name += '_'
        image_name += '.jpeg'
        gt_path = os.path.join(args.gt_dir, image_name)
        result.append(pool.apply_async(run, (image_path, gt_path, mask_path)))
    except:
        pass

pool.close()

for res in result:
  psnr, ssim = res.get()
  psnr_list.append(psnr)
  ssim_list.append(ssim)


p_m = np.mean(psnr_list)
psnr_diff =  np.std(psnr_list)

s_m = np.mean(ssim_list)
ssim_diff = np.std(ssim_list)

print('PSNR : {} +- {}'.format(p_m, psnr_diff))
print('SSIM : {} +- {}'.format(s_m, ssim_diff))