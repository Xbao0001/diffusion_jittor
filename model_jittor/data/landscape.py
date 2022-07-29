import os
from functools import partial

import albumentations as A
import cv2
import jittor as jt
import numpy as np
from jittor import transform
from jittor.dataset import Dataset
from PIL import Image

from .utils import *


class VQDataset(Dataset):
    def __init__(self, image_root: str, image_names: list, transform=None):
        super().__init__()
        self.image_root = image_root
        self.image_names = image_names
        self.transform = transform
        
        self.set_attrs(total_len=len(self.image_names))
    
    def __getitem__(self, idx):
        p_img = os.path.join(self.image_root, self.image_names[idx])
        image = cv2.imread(p_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        name = os.path.splitext(self.image_names[idx])[0] # remove ext (i.e. .png, .jpg)
        if image.shape[-1] ==  3:
            image = image.permute(2, 0, 1)
        return image, name
    

class LDMDataset(Dataset):
    def __init__(
            self, 
            image_root: str,
            segmentation_root: str,
            image_names: list,
            transform=None,
    ):
        super().__init__()
        self.image_root = image_root
        self.segmentation_root = segmentation_root
        self.image_names = image_names
        self.transform = transform

        self.set_attrs(total_len=len(self.image_names))

    def __getitem__(self, idx):
        path_img = os.path.join(self.image_root, self.image_names[idx])
        path_seg = os.path.join(self.segmentation_root, 
                                self.image_names[idx].replace('.jpg', '.png'))
        image = Image.open(path_img).convert('RGB')
        image = np.asarray(image)
        seg = Image.open(path_seg)
        seg = np.asarray(seg)
        if self.transform is not None:
            image, seg = self.transform(image=image, mask=seg).values()
        name = os.path.splitext(self.image_names[idx])[0] # remove ext (i.e. .png, .jpg)
        return image, seg, name


class InferenceDataset(Dataset):
    """ use for inference on val(test) dataset, resize (768x1024) to (384x512)
    """
    def __init__(self, segmentation_root: str, n_labels=29):
        super().__init__()
        self.segmentation_root = segmentation_root
        self.n_labels = n_labels
        self.segmentations = os.listdir(self.segmentation_root)

        self.resize = transform.Compose([
            transform.Resize((384, 512)),
            transform.ToTensor(),
        ])
        
        self.set_attrs(total_len=len(self.segmentations))
    
    def __getitem__(self, idx):
        file_name = self.segmentations[idx]
        name = os.path.splitext(file_name)[0]
        seg = Image.open(os.path.join(self.segmentation_root, file_name))

        seg = jt.array(self.resize(seg)).long()
        seg = jt.init.eye(self.n_labels)[seg]
        seg = seg.permute(0, 3, 1, 2).squeeze(0) 
        return seg, name


def get_vq_dataloader( 
    image_root='/nas/datasets/landscape/train_val/images', 
    train_val_split_ratio=0.9,
    train_val_split_seed=42,
    batch_size=8, 
    num_workers=2, 
    image_size=(256, 256),
):
    train_images, val_images = train_val_split(
        image_root=image_root, 
        ratio=train_val_split_ratio, 
        seed=train_val_split_seed, 
    )

    height, width = image_size 
    train_transform = A.Compose([ 
        A.RandomResizedCrop(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.Lambda(image=to_tensor),
    ])
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=height),
        A.RandomCrop(width=width, height=height),
        A.Lambda(image=to_tensor),
    ])

    train_loader = VQDataset(
        image_root, train_images, train_transform,
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = VQDataset(
        image_root, val_images, val_transform
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True,
    )
    return train_loader, val_loader

    
# TODO: add size support
def get_ldm_dataloader(
    image_root='/nas/datasets/landscape/train_val/images',
    segmentation_root='/nas/datasets/landscape/train_val/labels',
    train_val_split_ratio=0.9,
    train_val_split_seed=42,
    batch_size=6,
    num_workers=2,
    n_labels=29,
    image_size=(256, 256),
):
    
    train_images, val_images = train_val_split(
        image_root=image_root, 
        ratio=train_val_split_ratio, 
        seed=train_val_split_seed, 
    )

    height, width = image_size 
    train_transform = A.Compose([
        A.RandomResizedCrop(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.Lambda(image=to_tensor, 
                 mask=partial(to_onehot, n_labels=n_labels)),
    ])
    val_transform = A.Compose([ # use 384x512 directly?
        A.SmallestMaxSize(max_size=height),
        A.RandomCrop(width=width, height=height),
        A.Lambda(image=to_tensor,
                 mask=partial(to_onehot, n_labels=n_labels)),
    ])

    train_dataloader = LDMDataset(
        image_root=image_root,
        segmentation_root=segmentation_root,
        image_names=train_images,
        transform=train_transform,
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_dataloader = LDMDataset(
        image_root=image_root,
        segmentation_root=segmentation_root,
        image_names=val_images,
        transform=val_transform,
    ).set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True, 
    )
    return train_dataloader, val_dataloader
