# -*- coding: UTF-8 -*-
"""
  @Author  ： GongTao Yue
  @Date    ： 2024/3/20
"""

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
import random
import torch


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "train" if train else "val"
        data_root = os.path.join(root, "TNBC", self.flag)

        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.transform = transforms

        # 读取原始图像路径
        img_dir = os.path.join(data_root, "images")
        img_names = [i for i in os.listdir(img_dir) if i.endswith(".png")]
        self.img_paths = [os.path.join(img_dir, i) for i in img_names]
        # 读取mask图路径
        mask_dir = os.path.join(data_root, "masks")
        mask_names = [i for i in os.listdir(mask_dir) if i.endswith(".png")]
        self.mask_paths = [os.path.join(mask_dir, i) for i in mask_names]

        seg_prior_dir = os.path.join(data_root, "seg_priors")
        seg_prior_names = [i for i in os.listdir(seg_prior_dir) if i.endswith(".png")]
        self.seg_prior_paths = [os.path.join(seg_prior_dir, i) for i in seg_prior_names]

        boundary_dir = os.path.join(data_root, "boundary_priors")
        boundary_names = [i for i in os.listdir(boundary_dir) if i.endswith(".png")]
        self.boundary_paths = [os.path.join(boundary_dir, i) for i in boundary_names]

    def __getitem__(self, idx):
        # 读取原始图像
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 读取mask图
        mask = cv2.imread(self.mask_paths[idx])

        seg_prior_path = self.seg_prior_paths[idx]
        seg_prior = cv2.imread(seg_prior_path)
        boundary_path = self.boundary_paths[idx]
        boundary_prior = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)

        seed = random.randint(0, 2 ** 32)

        if self.transform is not None:
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            img = Image.fromarray(img)
            img = self.transform(img)

            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            mask = Image.fromarray(mask)
            mask = self.transform(mask)

            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            seg_prior = Image.fromarray(seg_prior)
            seg_prior = self.transform(seg_prior)

            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            boundary_prior = Image.fromarray(boundary_prior)
            boundary_prior = self.transform(boundary_prior)

        # 将mask转换为int64
        mask = np.array(mask)
        labels = mask[0, :, :]
        return img, seg_prior, boundary_prior, np.int64(labels)

    def __len__(self):
        return len(self.img_paths)
