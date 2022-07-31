import os
import shutil
import time
from functools import wraps
from typing import Optional, Union

import jittor as jt
import numpy as np
import wandb
from einops import rearrange
from PIL import Image


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def start_grad(model):
    for param in model.parameters():
        if 'running_mean' in param.name() or 'running_var' in param.name():
            continue
        param.start_grad()


def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()


def toggle_to_train(model):
    """set latent diffusion model to train mode"""
    # model.first_stage_model.eval() # already set to eval when initialized
    model.cond_stage_model.train()
    model.model.train()


def toggle_to_eval(model):
    """set latent diffusion model to eval mode"""
    # model.first_stage_model.eval()
    model.cond_stage_model.eval()
    model.model.eval()


def make_grid(images: Union[np.ndarray, jt.Var], n_cols=4):
    """ 
    convert a batch of numpy images to one image which has n_cols columns.

    Args:
        img: numpy.ndarray or jt.Var of shape (b c h w) or (b h w c)
        n_cols: number of images per row

    Returns:
        single image of shape (h*n_rows, w*n_cols, c), where n_rows * n_cols 
        = batch_size, if not, it will add some black images
    """
    if isinstance(images, jt.Var):
        images = images.detach().data
    assert len(images.shape) == 4, 'image must be 4d array'

    if images.shape[0] % n_cols != 0:
        B, C, H, W = images.shape
        padding_images = np.zeros((n_cols - (B % n_cols), C, H, W))
        images = np.vstack((images, padding_images))

    if images.shape[1] == 3 or images.shape[1] == 1:
        grid = rearrange(images,
                         '(n_rows n_cols) c h w -> (n_rows h) (n_cols w) c',
                         n_cols=n_cols)
    elif images.shape[3] == 3 or images.shape[3] == 1:
        grid = rearrange(images,
                         '(n_rows n_cols) h w c -> (n_rows h) (n_cols w) c',
                         n_cols=n_cols)
    else:
        raise ValueError(f"Can not process images of shape: {images.shape}")

    if grid.shape[-1] == 1:  # for wandb to log segmentation
        return grid[:, :, 0]
    return grid


def to_pil_image(x: Union[np.ndarray, jt.Var]):
    if isinstance(x, jt.Var): x = x.data
    assert x.ndim == 3, "only support C H W or H W C, where C = 3"
    if x.shape[0] == 3: x = x.transpose(1, 2, 0) # C H W -> H W C
    return Image.fromarray(np.uint8(x * 255), 'RGB')



def trace_time(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start_tick = time.time()
        res = func(*args, **kwargs)
        end_tick = time.time()
        if jt.rank == 0:
            if "epoch" in kwargs:
                print(f"Epoch {kwargs['epoch']:3d} ", end='')
            print(f"Elapse: {end_tick - start_tick/60:.2f} min.")
        return res
    return inner


def master_only(func):
    @wraps(func)
    def inner(*args, **kwargs):
        res = None
        if jt.rank == 0:
            res = func(*args, **kwargs)
        return res
    return inner


def format_var(x: Union[jt.Var, float, int, str]):
    if hasattr(x, "data"):
        return x.data[0]
    else:
        return x


@master_only
def log(info_dict: dict,
        epoch: int,
        step: int,
        global_step: int,
        total: int,
        stage: str = "train",
        wandb_enabled=True,
        images: Optional[dict] = None,
        image_interval=10,
        log_interval=10):
    if step % log_interval != 0:
        return
    print(f"[{stage}] Epoch: {epoch} [{step}/{total}]\t",
          *[f"{k}: {format_var(v):6.4f}  " for k, v in info_dict.items()])
    if wandb_enabled:
        wandb.log({
            f"{stage}/{k}": format_var(v) for k, v in info_dict.items()
        }, commit=False)
        wandb.log({
            f"{stage}/iter": global_step,
            f"{stage}/epoch": epoch
        })
        if images is not None and step % image_interval == 0:
            wandb.log({
                f"{stage}/{k}": wandb.Image(
                    make_grid(((v.clamp(-1, 1) + 1) / 2).data, 4)
                ) for k, v in images.items()
            })


@master_only
def save_checkpoint(checkpoint, save_dir, filename=None, is_best=False):
    file_path = os.path.join(save_dir, filename)
    jt.save(checkpoint, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(save_dir, 'best_model.ckpt'))
