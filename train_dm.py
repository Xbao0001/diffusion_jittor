import builtins
import os
import time

import hydra
import jittor as jt
import wandb
from omegaconf import OmegaConf

from model_jittor.data.cifar10 import get_cifar10_dataloader
from model_jittor.ldm.diffusion import GaussianDiffusion
from model_jittor.scheduler import LinearWarmupCosineAnnealingLR
from model_jittor.utils import (make_grid, save_checkpoint,
                                to_save_or_not_to_save)


@hydra.main(version_base=None, config_path='configs', config_name='cifar10.yaml')
def init_and_run(cfg):
    # init jittor
    jt.flags.use_cuda = cfg.jittor.use_cuda
    jt.set_global_seed(cfg.jittor.seed)

    # only print on the master node
    if jt.world_size > 1 and jt.rank != 0:
        def print_pass(*args): pass
        builtins.print = print_pass

    # configure wandb and ckpt's save dir
    if jt.rank == 0:
        cfg.save_dir = os.path.join(cfg.save_dir, cfg.name)
        os.makedirs(cfg.save_dir, exist_ok=True)
        if cfg.wandb:
            config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True)
            wandb.init(project=cfg.project, name=cfg.name,
                       config=config, dir=cfg.save_dir)
        cfg.save_dir = os.path.join(cfg.save_dir, 'checkpoints')
        os.makedirs(cfg.save_dir, exist_ok=True)
        print(f'Saving checkpoints in {cfg.save_dir}')

    # run
    main(cfg)


def main(cfg):
    # TODO: try pred x_0
    # dataloader
    train_loader, _ = get_cifar10_dataloader(**cfg.data)

    # init model
    model = GaussianDiffusion(**cfg.diffusion)

    # configure optimizer TODO: try lr_scheduler and different lr
    # try adamW
    optimizer = jt.optim.Adam(
        list(model.model.parameters()),
        lr=cfg.lr,
    )
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=cfg.warmup_epochs,
        T_max=cfg.epochs,
    )

    # resume
    if cfg.resume is not None:
        assert os.path.isfile(cfg.resume)
        checkpoint = jt.load(cfg.resume)
        cfg.start_epoch = checkpoint['epoch'] + 1  # start from the next epoch
        model.load_state_dict(checkpoint['model'])
        model.ema_param_dict = checkpoint['ema_model']
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Start training, good luck!')
    for epoch in range(cfg.start_epoch, cfg.epochs):
        start_time = time.time()
        lr_scheduler.step()
        
        model.train()
        for i, (img, _) in enumerate(train_loader):
            global_train_steps = epoch * len(train_loader) + i

            loss = model(img)
            optimizer.step(loss)

            jt.sync_all()

            if model.use_ema:
                model.step_ema()

            if global_train_steps % cfg.print_freq == 0:
                print(
                    f"epoch: {epoch:3d}\t",
                    # seems a bug from CIFAR10
                    f"iter: [{i:4d}/{len(train_loader) // cfg.data.batch_size}]\t",
                    f"loss {loss.item():7.3f}\t",
                )
                if jt.rank == 0 and cfg.wandb:
                    wandb.log({
                        "train/epoch": epoch,
                        "train/iter": global_train_steps,
                        "train/lr": optimizer.lr,
                        "train/loss": loss.item(),
                    })

        train_time = time.time() - start_time
        print(f'Epoch {epoch:3d} training time: {train_time/60:.2f} min.')

        # sample val set
        if epoch % cfg.sample_freq == 0:
            if jt.rank == 0 and cfg.wandb:
                model.eval()
                img_sample = model.ddpm_sample(shape=(36, 3, 32, 32))
                wandb.log({
                    'generated': wandb.Image(make_grid(img_sample.data, n_cols=6)),
                    'original': wandb.Image(make_grid(img.data[:36], n_cols=6)), # potential bug here, if batch size < 36
                })

        if to_save_or_not_to_save(epoch, cfg.epochs, cfg.save_freq):
            save_checkpoint({
                'epoch': epoch,
                'loss': loss.data,
                'model': model.state_dict(),
                'ema_model': model.ema_param_dict,
                'optimizer': optimizer.state_dict(),
            }, save_dir=cfg.save_dir, filename=f"epoch_{epoch}.ckpt")


if __name__ == "__main__":
    init_and_run()
