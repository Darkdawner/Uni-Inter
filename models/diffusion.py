import torch
import numpy as np
import math


def sqrt_beta_schedule(timesteps, s=0.0001):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = 1 - torch.sqrt(t + s)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3., end=3., tau=0.7, clamp_min=1e-5):
    """Sigmoid schedule as proposed in https://arxiv.org/abs/2212.11972"""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 motion_size=(120, 66), device="cuda", ddim_timesteps=100,
                 scheduler='Linear'):
        self.noise_steps = noise_steps
        self.beta_start = (1000 / noise_steps) * beta_start
        self.beta_end = (1000 / noise_steps) * beta_end
        self.motion_size = motion_size
        self.device = device
        self.scheduler = scheduler

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)
        self.ddim_timesteps = ddim_timesteps

        self.ddim_timestep_seq = np.asarray(
            list(range(0, self.noise_steps, self.noise_steps // self.ddim_timesteps))) + 1
        self.ddim_timestep_prev_seq = np.append(np.array([0]), self.ddim_timestep_seq[:-1])

    def prepare_noise_schedule(self):
        if self.scheduler == 'Linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.scheduler == 'Cosine':
            return cosine_beta_schedule(self.noise_steps)
        elif self.scheduler == 'Sqrt':
            return sqrt_beta_schedule(self.noise_steps)
        elif self.scheduler == 'Sigmoid':
            return sigmoid_beta_schedule(self.noise_steps)
        else:
            raise NotImplementedError(f"unknown scheduler: {self.scheduler}")

    def noise_motion(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None, None]
        noise = torch.randn_like(x).to(x.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_ddim_pred_x0_sum(self, model, cond_voxel_grid, text_features,
                                 sample_num, noise=None, cond_out=False):
        """DDIM sampling with direct x0 prediction."""
        model.eval()

        if noise is not None:
            x = noise
        else:
            x = torch.randn((
                sample_num, self.motion_size[0], self.motion_size[1],
                self.motion_size[2], self.motion_size[3], self.motion_size[4]
            )).to(self.device)

        with torch.no_grad():
            for i in reversed(range(0, self.ddim_timesteps)):
                t = (torch.ones(sample_num) * self.ddim_timestep_seq[i]).long().to(self.device)
                prev_t = (torch.ones(sample_num) * self.ddim_timestep_prev_seq[i]).long().to(self.device)

                alpha_hat = self.alpha_hat[t][:, None, None, None, None, None]
                alpha_hat_prev = self.alpha_hat[prev_t][:, None, None, None, None, None]

                if cond_voxel_grid is not None:
                    if cond_out:
                        predicted_x0, _ = model(x, cond_voxel_grid, text_features, t)
                    else:
                        predicted_x0 = model(x, cond_voxel_grid, text_features, t)
                else:
                    predicted_x0 = model(x, text_features, t)

                predicted_noise = (x - torch.sqrt(alpha_hat) * predicted_x0) / torch.sqrt(1. - alpha_hat)
                pred_dir_xt = torch.sqrt(1 - alpha_hat_prev) * predicted_noise
                x_prev = torch.sqrt(alpha_hat_prev) * predicted_x0 + pred_dir_xt
                x = x_prev

            return predicted_x0
