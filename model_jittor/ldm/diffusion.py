"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/07/07 19:06:07
@Description: 

Classes to be Implemented:
- [x] LatentDiffusion
- [x] DiffusionWrapper
"""


import os
from contextlib import contextmanager
from functools import partial

import jittor as jt
import jittor.nn as nn
import numpy as np
from tqdm import tqdm

import model_jittor.utils as utils
from ..autoencoder.vqgan import VQModelInference
from .diffusion_utils import default, extract, make_beta_schedule
from .ema import EMAModel
from .modules import SpatialRescaler
from .openai_model import UNetModel


class GaussianDiffusion(nn.Module):
    """main class"""

    def __init__(
        self,
        beta_schedule="cosine",
        timesteps=1000,
        image_size=64,
        channels=3,
        loss_type='l1',
        objective='pred_noise',
        # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_gamma=0.,
        p2_loss_weight_k=1,
        use_ema=False,
        load_ema_model=False,
        clip_denoised=True,
        unet_config=None,
        ckpt_path=None,
    ):
        super().__init__()

        self.image_size = image_size
        self.channels = channels
        self.loss_type = loss_type
        assert objective in ['pred_noise', 'pred_x0']
        self.objective = objective
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        self.p2_loss_weight_k = p2_loss_weight_k
        self.clip_denoised = clip_denoised
        self.use_ema = use_ema
        self.load_ema_model = load_ema_model
        
        self.register_schedule(beta_schedule, timesteps)

        self.model = UNetModel(**unet_config)
        if self.use_ema:
            print('Using ema')
            self.ema_model = EMAModel(self.model)

        if ckpt_path is not None:
            assert os.path.isfile(ckpt_path), f"Cannot find ckpt: {ckpt_path}"
            ckpt = jt.load(ckpt_path)
            if isinstance(ckpt, dict) and 'model' in ckpt.keys():
                self.load_state_dict(ckpt['model'])
                if self.use_ema and self.load_ema_model:
                    self.model.load_state_dict(ckpt['ema_model'])
                    print('Loaded ema model')
                print(f"Using checkpoint from epoch {ckpt['epoch']}")
            else:
                self.load_state_dict(ckpt)
            print(f"Loaded  ckeckpoint from '{ckpt_path}'")

    def register_buffer(self, name, attr):
        # TODO: print all params and check requires_grad
        if isinstance(attr, jt.Var):
            attr.stop_grad()
        setattr(self, name, attr)

    def register_schedule(self, beta_schedule, timesteps):
        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = len(betas)
        assert alphas_cumprod.shape[0] == self.num_timesteps

        to_var = partial(jt.array, dtype=jt.float32)

        self.register_buffer('betas', to_var(betas))
        self.register_buffer('alphas_cumprod', to_var(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_var(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_var(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_var(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_var(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_var(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_var(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_var(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_var(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_var(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_var(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('p2_loss_weight', to_var((self.p2_loss_weight_k + alphas_cumprod / 
                                                (1 - alphas_cumprod)) ** -self.p2_loss_weight_gamma))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t):
        model_output = self.model(x, t)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised: bool):
        _, x_start = self.model_predictions(x, t)

        if clip_denoised:
            x_start = jt.clamp(x_start, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @jt.no_grad()
    def p_sample(self, x, t: int, clip_denoised: bool):
        batched_times = jt.full(
            (x.shape[0],), t, dtype=jt.int64)  # NOTE: why 64?
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = jt.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        return model_mean + (0.5 * model_log_variance).exp() * noise

    @jt.no_grad()
    def p_sample_loop(self, shape, x_T=None, clip_denoised=None):
        clip_denoised = default(clip_denoised, self.clip_denoised)
        img = default(x_T, jt.randn(shape))

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='DDPM sampling'):
            img = self.p_sample(img, t, clip_denoised)

        img = utils.unnormalize_to_zero_to_one(img)
        return img

    @jt.no_grad()
    def ddim_sample(self, shape, sampling_steps, eta=1., x_T=None, clip_denoised=None):
        clip_denoised = default(clip_denoised, self.clip_denoised)
        times = jt.linspace(0., self.num_timesteps,
                            steps=sampling_steps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = default(x_T, jt.randn(shape))
        assert not img.requires_grad  # TODO

        for time, time_next in tqdm(time_pairs, desc='DDIM sampling'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = jt.full((shape[0],), time,
                                dtype=jt.int64)  # NOTE: why 64

            pred_noise, x_start = self.model_predictions(img, time_cond)

            if clip_denoised:
                x_start = jt.clamp(x_start, -1., 1.)

            sigma = eta * ((1 - alpha / alpha_next) *
                           (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = jt.randn_like(img) if time_next > 0 else 0.

            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        img = utils.unnormalize_to_zero_to_one(img)
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, jt.randn_like(x_start))
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, jt.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        output = self.model(x_noisy, t)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if self.loss_type == 'l1':
            loss = (output - target).abs()
        elif self.loss_type == 'l2':
            loss = (output - target).sqr()
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape) 
        return loss.mean()

    def execute(self, img, *args, **kwargs):
        t = jt.randint(0, self.num_timesteps, (img.shape[0],)).long()
        img = utils.normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


class LatentDiffusion(GaussianDiffusion):
    """main class"""

    def __init__(
        self,
        first_stage_config=None,
        cond_stage_config=None,
        # use_ema=True,  # TODO
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # self.use_ema = use_ema

        self.first_stage_model = VQModelInference(**first_stage_config)
        self.first_stage_model.eval()

        self.cond_stage_model = SpatialRescaler(**cond_stage_config)
        self.model = DiffusionWrapper(self.model)
        # if self.use_ema:
        #     self.ema_model = EMAModel(self.model)
        #     self.ema_model.register()

    # def step_ema(self):
    #     if self.use_ema:
    #         self.ema_model.step()

    # @contextmanager
    # def ema_scope(self, enable=False, context=None):
    #     if self.use_ema and enable:
    #         self.ema_model.apply_shadow()
    #         if context is not None:
    #             print(f"{context}: Switched to EMA weights")
    #     try:
    #         yield None
    #     finally:
    #         if self.use_ema and enable:
    #             self.ema_model.restore()
    #             if context is not None:
    #                 print(f"{context}: Restored training weights")

    def execute(self, x, c, return_encoded=False, *args, **kwargs):
        x = self.encode_first_stage(x).detach()  # B 3 H W -> B C H//4 W//4

        t = jt.randint(0, self.num_timesteps, (x.shape[0],)).long()
        c = self.cond_stage_model(c)  # B 29 H W -> B 3 H//4 W//4
        # FIXME: norm to -1, 1
        loss = self.p_losses(x, c, t, *args, **kwargs)
        if return_encoded:
            return loss, x.detach()
        else:
            return loss

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: jt.randn_like(x_start).stop_grad())
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(x_noisy, t, cond)
        loss_simple = self.loss_fn(model_output, noise)
        return loss_simple

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: jt.randn_like(x_start).stop_grad())
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return nn.l1_loss
        elif self.loss_type == 'l2':
            return nn.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    @jt.no_grad()
    def sample_and_decode(self, cond):
        img_latent = self.sample(cond)
        return self.decode_first_stage(img_latent)

    @jt.no_grad()
    def sample(self, cond, return_intermediates=False,
               quantize_denoised=False, log_every_t=None):
        cond = self.cond_stage_model(cond)  # B 29 H W -> B 3 H//4 W//4
        # shape = (cond.shape[0], self.channels, self.image_size, self.image_size)
        B, _, h, w = cond.shape
        shape = (B, self.channels, h, w)

        return self.p_sample_loop(cond, shape, return_intermediates,
                                  quantize_denoised=quantize_denoised,
                                  log_every_t=log_every_t)

    @jt.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      quantize_denoised=False, log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        b = shape[0]
        img = jt.randn(shape).stop_grad()
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            ts = jt.full((b,), i, dtype=jt.int64)
            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if i % log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    @jt.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 quantize_denoised=False, return_x0=False, temperature=1.):
        b, *_, _ = *x.shape, 0
        out = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                   quantize_denoised=quantize_denoised,
                                   return_x0=return_x0)
        if return_x0:
            model_mean, _, model_log_variance, x0 = out
        else:
            model_mean, _, model_log_variance = out

        noise = noise_like(x.shape, repeat_noise) * temperature
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, c, t, clip_denoised, quantize_denoised=False,
                        return_x0=False):
        model_out = self.model(x, t, c)
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

        if clip_denoised:
            x_recon = jt.clamp(x_recon, -1., 1.)
        if quantize_denoised:
            x_recon, _, _ = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    @jt.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z)

    @jt.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


class DiffusionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.diffusion_model = model

    def execute(self, x, t, c_concat):
        xc = jt.concat([x, c_concat], dim=1)
        out = self.diffusion_model(xc, t)

        return out
