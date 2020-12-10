import os

import numpy as np
import torch
import random
from PIL import Image
from torch.utils import data

num_classes = 19
ignore_label = 255
root = '/home/akumar/dataset/games_dataset'

img_mean = np.array([103.939, 116.779, 123.68]) / 255.


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

# Zero pad the palette
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


# Colorize the mask accordint to cityscapes
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask



def dataset_generator(mode):

    mask_path = os.path.join(root, mode, 'labels')
    mask_postfix = '.png'
    img_path = os.path.join(root, mode, 'images')
    assert os.listdir(img_path) == os.listdir(mask_path)
    items = []

    c_items = [name.split('.png')[0] for name in os.listdir(img_path)]

    if mode == 'train':
        for it in c_items[:5000]:
            item = (os.path.join(img_path, it + '.png'), os.path.join(mask_path, it + mask_postfix))
            items.append(item)
    elif mode == 'val' or mode == 'test':
        for it in c_items[:1000]:
            item = (os.path.join(img_path, it + '.png'), os.path.join(mask_path, it + mask_postfix))
            items.append(item)
    return items


class CityScapes(data.Dataset):
    def __init__(self, mode, joint_transform=None, transform=None, target_transform=None):
    # def __init__(self, mode, transform=None):

        self.imgs = dataset_generator(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.img_mean = img_mean
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18, 34:ignore_label}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)


        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))
        
        if self.mode == 'train':
            if self.transform is not None:

                seed = random.randint(0, 64)
                random.seed(seed)
                torch.manual_seed(seed)

                img = self.transform(img)
                
                random.seed(seed)
                torch.manual_seed(seed)
                mask = self.transform(mask)

            img = np.asarray(img)
            mask = np.asarray(mask)
            
            # reduce mean
            img = img[:, :, ::-1]  # switch to BGR
            img = np.transpose(img, (2, 0, 1)) / 255.
            img[0] -= self.img_mean[0]
            img[1] -= self.img_mean[1]
            img[2] -= self.img_mean[2]

            # convert to tensor
            img = torch.from_numpy(img.copy()).float()
            mask = torch.from_numpy(mask.copy()).long()

        else:
            if self.joint_transform is not None:
                img, mask = self.joint_transform(img, mask)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                mask = self.target_transform(mask)


        if self.mode == 'test':
            return img_path, img, mask
        else:
            return img, mask

    def __len__(self):
        return len(self.imgs)
