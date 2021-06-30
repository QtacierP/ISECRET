# -*- coding: utf-8 -*-
from builtins import print
import torch
from random import randint
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import os
import glob
import random
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from copy import deepcopy



class CustomDataSet(Dataset):
    '''
    A fundamental class for image-directory loader dataset
    '''
    def __init__(self, args, main_dir, transform,
                 need_name=False, need_shape=False, mask_dir=None, mask_transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.step = args.train.step
        self.need_name = need_name
        self.need_shape = need_shape
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)
        if args.train.len != 0:
            self.len = args.train.len
        else:
            self.len = len(self.total_imgs)
        self.need_mask = False if mask_transform is None else True
        self.mask_dir = mask_dir
        self.mask_transform = mask_transform

    def __len__(self):
        return self.len * self.step

    def __getitem__(self, idx, seed=None):
        img_id = self.total_imgs[idx % len(self.total_imgs)]
        img_loc = os.path.join(self.main_dir, img_id)
        image = Image.open(img_loc)
        if seed is None:
            seed = random.randint(0, 9999999)
        torch.manual_seed(seed)
        tensor_image = self.transform(image)
        batch = {}
        if seed is not None:
            torch.manual_seed(seed)
        batch['image'] = tensor_image
        if self.need_name:
            batch['name'] = img_loc
        if self.need_shape:
            batch['shape'] = np.asarray(np.asarray(image).shape)
        return batch


class CustomDataSetGT(Dataset):
    '''
    A fundamental class for image-directory loader dataset with ground-truth
    '''
    def __init__(self, args, main_dir, gt_dir, transform,
                 need_name=False, need_shape=False, mask_dir=None, mask_transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.gt_dir = gt_dir
        self.step = args.train.step
        self.need_name = need_name
        self.need_shape = need_shape
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)
        all_gts = os.listdir(gt_dir)
        self.total_gts = sorted(all_gts)
        assert len(self.total_imgs) % len(self.total_gts) == 0
        self.pair_num = len(self.total_imgs) // len(self.total_gts)
        if args.train.len != 0:
            self.len = args.train.len
        else:
            self.len = len(self.total_imgs)
        self.need_mask = False if mask_transform is None else True
        self.mask_dir = mask_dir
        self.mask_transform = mask_transform

    def __len__(self):
        return self.len * self.step

    def __getitem__(self, idx, seed=None):
        img_id =  self.total_imgs[idx % len(self.total_imgs)]
        img_loc = os.path.join(self.main_dir, img_id)
        gt_loc = os.path.join(self.gt_dir, self.total_gts[idx // self.pair_num % len(self.total_gts)])
        image = Image.open(img_loc)
        gt = Image.open(gt_loc)
        if seed is not None:
            torch.manual_seed(seed)
        tensor_image = self.transform(image)
        batch = {}
        if seed is not None:
            torch.manual_seed(seed)
        tensor_gt = self.transform(gt)
        batch['good'] = tensor_gt
        batch['bad'] = tensor_image
        if self.need_name:
            batch['name'] = img_loc
        if self.need_shape:
            batch['shape'] = np.asarray(np.asarray(image).shape)
        return batch


class OfflineDegradeCustomDataSet(Dataset):
    '''
      A fundamental class for image-directory loader dataset with degrading
      The degrade operation must be offline.
      Given the image xx.jpg, we have several transformations, it should be
      - xx_yyy.jpg (yyy can represents any words)
    '''
    def  __init__(self, args, main_dir, degrade_dir, transform, mask_dir=None, mask_transform=None):
        self.main_dir = main_dir
        self.degrade_dir = degrade_dir
        self.transform = transform
        self.step = args.train.step
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)
        self._init_map()
        if args.train.len != 0:
            self.len = args.train.len
        else:
            self.len = len(self.total_imgs)
        self.need_mask = False if mask_transform is None else True
        self.mask_dir = mask_dir
        self.mask_transform = mask_transform

    def _init_map(self):
        self.degrade_map = {}
        for img_path in self.total_imgs:
            self.degrade_map[img_path] = []
        all_imgs = os.listdir(self.degrade_dir)
        total_imgs = sorted(all_imgs)
        for img_path in total_imgs:
            _, img_name = os.path.split(img_path)
            image_id_split = img_name.split('_')[:-1]
            image_id = ''
            for idx, word in enumerate(image_id_split):
                image_id += str(word)
                if idx != len(image_id_split) - 1:
                    image_id += '_'
            image_id += '.jpeg'
            self.degrade_map[image_id].append(img_path)

    def __len__(self):
        return self.len * self.step

    def __getitem__(self, idx, seed=None):
        name = self.total_imgs[idx % len(self.total_imgs)]
        img_loc = os.path.join(self.main_dir, name)
        image = Image.open(img_loc)
        noise_image_list = self.degrade_map[name]
        r =  randint(0, len(noise_image_list) - 1) # Choose degraded image randomly
        noise_loc = os.path.join(self.degrade_dir, noise_image_list[r])
        noise_image = Image.open(noise_loc)
        if seed is None:
            seed = random.randint(0, 9999999999)
        torch.manual_seed(seed)
        random.seed(seed)
        tensor_image = self.transform(image)
        torch.manual_seed(seed)
        random.seed(seed)
        noise_tensor_image = self.transform(noise_image)
        batch = {'image': tensor_image, 'noise': noise_tensor_image}
        return batch

class BaseDataset(Dataset):
    '''
    A fundamental class for unpaired dataset
    '''
    def __init__(self, args, good_dataset=None, bad_dataset=None):

        self.args = args
        if good_dataset is None or bad_dataset is None:
            self._init_dataset()
        self.good_data = good_dataset
        self.bad_data = bad_dataset
        self.N = max(self.good_data.__len__(), self.bad_data.__len__())
        
    def _init_dataset(self):
        pass

    def __getitem__(self, index):
        bad_batch = self.bad_data[index % len(self.bad_data)]
        bad_img = bad_batch['image']
        random_index = randint(0, self.N - 1)
        good_batch = self.good_data[random_index % len(self.good_data)]
        good_img = good_batch['image']
        good_noise_img = good_batch['noise']
        batch = {'good': good_img, 'noise_good': good_noise_img,
                'bad': bad_img}
        return batch

    def __len__(self):
        return self.N