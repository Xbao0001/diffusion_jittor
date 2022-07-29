import os
import random

import jittor as jt
import numpy as np


def to_tensor(img: np.ndarray, **kwargs):
    """[0, 255] -> [0, 1] """
    img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)
    return jt.array(img.transpose(2, 0, 1)).float32()


def to_onehot(mask: np.ndarray, n_labels=29, **kwargs):
    mask_onehot = np.eye(n_labels)[mask]
    return jt.array(mask_onehot.transpose(2, 0, 1)).float32()


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
