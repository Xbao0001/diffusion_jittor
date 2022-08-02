import os
import sys; sys.path.append('..')

import jittor as jt
from omegaconf import OmegaConf
from tqdm import tqdm

from model_jittor.ldm.diffusion import GaussianDiffusion
from model_jittor.ldm.model import Model

from model_jittor import utils


jt.flags.use_cuda = True


model = Model(resolution=32,
              in_channels=3,
              out_ch=3,
              ch=128,
              ch_mult=(1,2,2,2),
              num_res_blocks=2,
              attn_resolutions=(16,),
              dropout=0.1)
ckpt = jt.load('./ckpts/cifar10/ema_model.pkl')
model.load_state_dict(ckpt)
config = OmegaConf.load('./configs/cifar10.yaml')
diffusion = GaussianDiffusion(**config.diffusion)
diffusion.model = model

batch_size = 200
num_iter = 250
save_dir = './results/ddpm3'
os.makedirs(save_dir, exist_ok=True)

idx = 0

for i in tqdm(range(num_iter)):
    imgs = diffusion.p_sample_loop((batch_size, 3, 32, 32))
    for img in imgs:
        utils.to_pil_image(img).save(f"{save_dir}/img0_{idx:06d}.jpg")
        idx += 1
