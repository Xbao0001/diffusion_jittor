import argparse
import os

import jittor as jt
from omegaconf import OmegaConf
from tqdm import tqdm

from model_jittor import utils
from model_jittor.data.noise import NoiseDataset
from model_jittor.ldm.diffusion import GaussianDiffusion
from model_jittor.ldm.model import Model


def main(args, cfg):
    dataloader = NoiseDataset(length=args.num).set_attrs(
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    diffusion = GaussianDiffusion(**cfg.diffusion)

    if args.use_pretrained:
        model = Model(resolution=32,
                      in_channels=3,
                      out_ch=3,
                      ch=128,
                      ch_mult=(1, 2, 2, 2),
                      num_res_blocks=2,
                      attn_resolutions=(16,),
                      dropout=0.1)
        ckpt = jt.load(args.pretrained_ckpt)
        model.load_state_dict(ckpt)
        diffusion.model = model

    diffusion.eval()
    for noise, ids in tqdm(dataloader):
        if args.ddim:
            imgs = diffusion.ddim_sample(
                noise.shape, args.steps, args.eta, noise)
        else:
            imgs = diffusion.ddpm_sample(shape=noise.shape, x_T=noise)

        for img, idx in zip(imgs, ids):
            utils.to_pil_image(img).save(
                f"{args.output_path}/{args.prefix}_{idx:08d}.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/cifar10.yaml')
    parser.add_argument('--output_path', type=str, default='./results')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--prefix', type=str, default='img')

    parser.add_argument('--num', type=int, default=50000,
                        help='total num of images to generate')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=5)

    parser.add_argument('--ckpt_path', type=str, default=None, required=True)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str,
                        default='./save/ckpts/cifar10/ema_model.pkl')

    parser.add_argument('--ddim', action='store_true')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--eta', type=float, default=1.0)

    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.diffusion.ckpt_path = args.ckpt_path
    cfg.diffusion.load_ema_model = args.ema

    jt.flags.use_cuda = True
    if args.ema:
        args.name = args.name + '_ema'
    if args.seed is not None:
        jt.set_global_seed(args.seed)
        args.name = args.name + '_seed_' + str(args.seed)
    args.output_path = os.path.join(args.output_path, args.name)
    os.makedirs(args.output_path, exist_ok=True)
    print(f'Saving results in "{args.output_path}"')

    main(args, cfg)
