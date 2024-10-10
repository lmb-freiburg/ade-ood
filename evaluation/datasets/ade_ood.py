import os
from PIL import Image
import numpy as np
import torch

class ADEOoDDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.ood_idx = 1
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        self.gt_dir = os.path.join(root, 'annotations')
        self.img_names = os.listdir(self.img_dir)
        self.img_paths = [os.path.join(self.img_dir, img_name) for img_name in self.img_names]
        self.gt_paths = [os.path.join(self.gt_dir, img_name.replace(".jpg", "_mask.png")) for img_name in self.img_names]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        gt = np.array(Image.open(gt_path))
        return img, gt, img_name
