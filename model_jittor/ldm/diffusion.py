import os
from functools import partial

import jittor as jt
import jittor.nn as nn
import numpy as np
from tqdm import tqdm

import model_jittor.utils as utils
from ..autoencoder.vqgan import VQModelInference
from .diffusion_utils import default, extract, make_beta_schedule
from .modules import SpatialRescaler
from .openai_model import UNetModel


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        beta_schedule="cosine",
        timesteps=1000,
        image_size=64,  # TODO
        channels=3,
        loss_type='l1',
        objective='pred_noise',
        clip_denoised=True,
        # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_gamma=0.,
        p2_loss_weight_k=1,
        unet_config=None,
        use_ema=False,
        decay=0.9999,  # used for ema
        ckpt_path=None,
        load_ema_model=False,
    ):
        super().__init__()

        self.loss_type = loss_type
        assert objective in ['pred_noise', 'pred_x0']
        self.objective = objective
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        self.p2_loss_weight_k = p2_loss_weight_k
        self.clip_denoised = clip_denoised
        self.use_ema = use_ema
        self.decay = decay
        self.load_ema_model = load_ema_model
        self.unet_config = unet_config
        self.ema_param_dict = {}

        self.register_schedule(beta_schedule, timesteps)
        self.init_model()  # init model and ema

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def init_model(self):
        self.model = UNetModel(**self.unet_config)

        if self.use_ema:
            self.num_updates = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:  # in fact, this is always true
                    self.ema_param_dict[name] = param.detach().clone()

    def step_ema(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) /
                    (10 + self.num_updates))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_param_dict
                new_average = (1.0 - decay) * param.detach().clone() + \
                    decay * self.ema_param_dict[name]
                self.ema_param_dict[name] = new_average

    def load_checkpoint(self, ckpt_path):
        assert os.path.isfile(ckpt_path), f"Cannot find ckpt: {ckpt_path}"
        ckpt = jt.load(ckpt_path)
        if isinstance(ckpt, dict) and 'model' in ckpt.keys():
            self.load_state_dict(ckpt['model'])
            if self.use_ema and self.load_ema_model:
                assert 'ema_model' in ckpt.keys()
                self.model.load_state_dict(ckpt['ema_model'])
                print('Loaded ema model')
            print(
                f"Loaded checkpoint '{ckpt_path}' from epoch {ckpt['epoch']}")
        else:  # never use
            self.load_state_dict(ckpt)
            print(f"Loaded  ckeckpoint from '{ckpt_path}'")

    def register_buffer(self, name, attr):
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
        # below: log calculation clipped because the posterior variance is 0
        # at the beginning of the diffusion chain
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
        noise = jt.randn_like(x) if t > 0 else jt.zeros_like(x) # no noise if t == 0
        t = jt.full((x.shape[0],), t, dtype=jt.int64)  # NOTE: why 64?
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised)
        return model_mean + (0.5 * model_log_variance).exp() * noise

    @jt.no_grad()
    def ddpm_sample(self, shape=None, x_T=None, clip_denoised=None):
        img = default(x_T, jt.randn(shape))

        clip_denoised = default(clip_denoised, self.clip_denoised)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='DDPM sampling'):
            img = self.p_sample(img, t, clip_denoised)

        img = utils.unnormalize_to_zero_to_one(img)
        return img

    @jt.no_grad()
    def ddim_sample(self, shape, ddim_steps, eta=1., x_T=None, clip_denoised=None):
        clip_denoised = default(clip_denoised, self.clip_denoised)
        times = jt.linspace(0., self.num_timesteps, steps=ddim_steps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = default(x_T, jt.randn(shape))

        for time, time_next in tqdm(time_pairs, desc='DDIM sampling'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = jt.full((shape[0],), time, dtype=jt.int64)

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
        noise = default(noise, jt.randn_like(x_start)).stop_grad()

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


# TODO: add classifier free guidence
class LatentDiffusion(GaussianDiffusion):
    """main class"""

    def __init__(
        self,
        first_stage_config=None,
        cond_stage_config=None,
        *args, **kwargs,
    ):
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config

        super().__init__(*args, **kwargs)

    def init_model(self):
        self.vqgan = VQModelInference(**self.first_stage_config)
        self.vqgan.eval()

        self.model = LDMWrapper(self.unet_config, self.cond_stage_config)

        if self.use_ema:
            self.ema_param_dict = {}
            self.num_updates = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:  # in fact, this is always true
                    self.ema_param_dict[name] = param.detach().clone()

    def execute(self, img, seg, *args, **kwargs):
        # seg: use segmentation(seg) as condition(cond)
        img = self.vqgan.encode(img).detach()  # B 3 H W -> B C H//4 W//4
        # rescale seg in self.model, i.e. LDMWrapper

        t = jt.randint(0, self.num_timesteps, (img.shape[0],)).long()
        # TODO: test whether to norm to -1, 1 ?
        # img = utils.normalize_to_neg_one_to_one(img)
        return self.p_losses(img, seg, t, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, jt.randn_like(x_start)).stop_grad()
        assert noise.requires_grad == False  # TODO: remove
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        output = self.model(x_noisy, t, cond)
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

    @jt.no_grad()
    def sample_and_decode(
        self, 
        seg, 
        shape=None, 
        x_T=None,
        clip_denoised=None,
        quantize_denoised=None,
        use_ddim=False,
        ddim_steps=100,
        eta=1.,
    ):
        if use_ddim:
            img_latent = self.ddim_sample(seg, shape, ddim_steps, eta, x_T, 
                                          clip_denoised, quantize_denoised)
        else:
            img_latent = self.ddpm_sample(seg, shape, x_T, clip_denoised, 
                                          quantize_denoised)
        img_decode = self.vqgan.decode(img_latent)
        return utils.unnormalize_to_zero_to_one(img_decode)

    @jt.no_grad()
    def ddpm_sample(self, cond, shape=None, x_T=None, clip_denoised=None,
                    quantize_denoised=False, ):
        img = default(x_T, jt.randn(shape))

        clip_denoised = default(clip_denoised, self.clip_denoised)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='DDPM sampling'):
            img = self.p_sample(img, cond, t, clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
        return img

    @jt.no_grad()
    def ddim_sample(self, seg, shape, ddim_steps, eta=1., x_T=None, clip_denoised=None):
        clip_denoised = default(clip_denoised, self.clip_denoised)
        times = jt.linspace(0., self.num_timesteps, steps=ddim_steps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = default(x_T, jt.randn(shape))

        for time, time_next in tqdm(time_pairs, desc='DDIM sampling'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            t = jt.full((shape[0],), time, dtype=jt.int64)

            pred_noise, x_start = self.model_predictions(img, t, seg)

            if clip_denoised:
                x_start = jt.clamp(x_start, -1., 1.)

            sigma = eta * ((1 - alpha / alpha_next) *
                           (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = jt.randn_like(img) if time_next > 0 else 0.

            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        return img

    @jt.no_grad()
    def p_sample(self, x, cond, t: int, clip_denoised=False, quantize_denoised=False):
        noise = jt.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        t = jt.full((x.shape[0],), t, dtype=jt.int64)
        model_mean, _, model_log_variance = self.p_mean_variance(
            x, cond, t, clip_denoised, quantize_denoised)
        return model_mean + (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, cond, t, clip_denoised, quantize_denoised=False):
        _, x_start = self.model_predictions(x, t, cond)
        if clip_denoised:
            x_start = jt.clamp(x_start, -1., 1.)
        if quantize_denoised:
            x_start, _, _ = self.vqgan.quantize(x_start)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def model_predictions(self, x, t, cond):
        model_output = self.model(x, t, cond)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return pred_noise, x_start


class LDMWrapper(nn.Module):
    def __init__(self, model, cond_stage_config):
        super().__init__()
        self.diffusion_model = model
        self.spatial_rescale = SpatialRescaler(**cond_stage_config)

    def execute(self, x, t, seg):
        seg = self.spatial_rescale(seg)  # B 29 H W -> B C2 H//4 W//4
        assert seg.shape[-2:] == x.shape[-2:]
        xc = jt.concat([x, seg], dim=1)
        out = self.diffusion_model(xc, t)
        return out
