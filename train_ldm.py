import builtins
import os
import time
import json

import hydra
import jittor as jt
import wandb
from omegaconf import OmegaConf

from model_jittor.data.landscape import get_ldm_dataloader
from model_jittor.ldm.diffusion import LatentDiffusion
from model_jittor import utils
from model_jittor.utils import make_grid


with open('./assets/class_labels.json', 'r') as f:
    _class_labels = json.load(f)
    class_labels = {}
    for key, val in _class_labels.items():
        class_labels[int(key)] = val


@hydra.main(version_base=None, config_path='configs', config_name='ldm.yaml')
def init_and_run(cfg):
    # init jittor
    jt.flags.use_cuda = cfg.jittor.use_cuda
    jt.set_global_seed(cfg.jittor.seed)

    # only print on the master node
    if jt.world_size > 1 and jt.rank != 0:
        def print_pass(*args): pass
        builtins.print = print_pass

    # TODO: adjust batch size and lr according to total batch size
    # if jt.world_size > 1:
    #     batch_size_old = cfg.data.batch_size
    #     cfg.data.batch_size *= jt.world_size
    #     cfg.lr *= jt.world_size
    #     print(f"Adjust batch size: {batch_size_old} -> {cfg.data.batch_size}")
    #     print(f"Adjust lr: {cfg.lr} -> {cfg.lr * jt.world_size}")

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

    main(cfg)


def main(cfg):
    # dataloader
    train_loader, val_loader = get_ldm_dataloader(**cfg.data)

    # init model and ema model
    model = LatentDiffusion(**cfg.model)

    # configure optimizer TODO: try lr_scheduler
    optimizer = jt.optim.AdamW(
        model.model.parameters(), 
        lr=cfg.lr,
        weight_decay=0,
    )

    # resume
    # TODO: update
    if cfg.resume is not None:
        assert os.path.isfile(cfg.resume)
        checkpoint = jt.load(cfg.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cfg.start_epoch = checkpoint['epoch'] + 1  # start from the next epoch

    print('Start training, good luck!')
    for epoch in range(cfg.start_epoch, cfg.epochs):
        start_time = time.time()
        model.model.train()
        for i, (img, seg, _) in enumerate(train_loader):
            global_train_steps = epoch * len(train_loader) + i

            img = utils.normalize_to_neg_one_to_one(img)
            loss = model(img, seg)
            optimizer.step(loss)

            jt.sync_all()
            if model.use_ema:
                model.step_ema() 

            if global_train_steps % cfg.print_freq == 0:
                print(
                    f"epoch: {epoch:3d}\t", 
                    f"iter: [{i:4d}/{len(train_loader)}]\t", 
                    f"loss {loss.item():7.3f}\t",                     
                )
                if jt.rank == 0: # TODO: warp wandb.log to master only
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
            model.model.eval()
            img, seg, _ = next(iter(val_loader))
            img, seg = img[:4], seg[:4]

            img_sample = model.sample_and_decode(seg)

            img_rec = model.vqgan(utils.normalize_to_neg_one_to_one(img))
            img_rec = utils.normalize_to_neg_one_to_one(img_rec)
            seg, _ = jt.argmax(seg.detach(), dim=1, keepdims=True)
            if jt.rank == 0 and cfg.wandb:
                wandb.log({
                    'generated': wandb.Image(make_grid(img_sample.data, n_cols=4)),
                    'reconstructed': wandb.Image(make_grid(img_rec.data, n_cols=4)),
                    'original': wandb.Image(
                        make_grid(img.data, n_cols=4), masks={
                            "ground_truth": 
                                {"mask_data": make_grid(seg.data, n_cols=4), 
                                "class_labels": class_labels}
                            },
                    ),
                })

        if utils.to_save_or_not_to_save(epoch, cfg.epochs, cfg.save_freq):
            utils.save_checkpoint({
                'epoch': epoch,
                'loss': loss.data,
                'model': model.state_dict(),
                'ema_model': model.ema_param_dict,
                'optimizer': optimizer.state_dict(),
            }, save_dir=cfg.save_dir, filename=f"epoch_{epoch}.ckpt")


if __name__ == "__main__":
    init_and_run()
