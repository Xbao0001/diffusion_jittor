import os
import random

import jittor as jt
import numpy as np
from albumentations import BasicTransform



def convert_to_negetive_one_positive_one(x: np.ndarray, **kwargs):
    """[0, 255] -> [-1, 1] """
    return x / 127.5 - 1.0


def to_onehot(mask: np.ndarray, n_labels=29, **kwargs):
    return np.eye(n_labels)[mask]


class ToVar(BasicTransform):
    def __init__(self, transpose_mask=True, always_apply=True, p=1.0):
        super(ToVar, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, "masks": self.apply_to_masks}

    def apply(self, img, **params):  
        if len(img.shape) not in [2, 3]:
            raise ValueError(
                "Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return jt.array(img.transpose(2, 0, 1)).float32()

    def apply_to_mask(self, mask, **params):  
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return jt.array(mask).float32()

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}


def train_val_split(image_root, ratio: float, seed: int, save_dir='./assets'):
    """split dataset to train and val

    Args:
        image_root (str): path to train_val images
        ratio (float, optional): train val split ratio. Defaults to 0.9.
        seed (int, optional): split seed. Defaults to 42.
        save (bool): whether to save the split results. Defaults to './assets'

    Returns:
        tuple: two lists of train and val image names
    """
    assert os.path.isdir(image_root)
    image_names =  os.listdir(image_root)
    random.seed(seed)
    random.shuffle(image_names)
    train_length = int(ratio * len(image_names))
    print(f"There are total {len(image_names)} images, ", 
          f"use {train_length} images for training and ", 
          f"{len(image_names) - train_length} images for validation.")
    train_images = image_names[:train_length]
    val_images = image_names[train_length:]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/train.txt', 'w') as file:
            for i in train_images:
                file.write(i + '\n')
        with open(f'{save_dir}/val.txt', 'w') as file:
            for i in val_images:
                file.write(i + '\n')
        print('save train and val image names in train.txt and val.txt')
    return train_images, val_images
