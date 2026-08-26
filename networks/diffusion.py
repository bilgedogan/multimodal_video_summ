import math

import torch
from torch import nn


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the diffusion timestep."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


class DiffusionDenoiser(nn.Module):
    """DDPM noise predictor over per-frame fused feature vectors.
    """

    def __init__(self, dim, hidden_dim, num_steps, time_dim=128, sdedit_t=5, sdedit_seed=0):
        super().__init__()
        self.num_steps = num_steps
        self.time_dim = time_dim
        self.sdedit_t = int(max(0, min(sdedit_t, num_steps - 1)))
        self.sdedit_seed = sdedit_seed

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.mid = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)

        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_acp', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_acp', (1.0 - alphas_cumprod).sqrt())

    def forward(self, x_t, t):
        """Predict the noise in x_t (N x dim) at timestep t (N,)."""
        h = self.in_proj(x_t) + self.time_mlp(timestep_embedding(t, self.time_dim))
        h = self.mid(h)
        return self.out_proj(self.out_norm(h))

    def q_sample(self, x0, t, noise):
        return self.sqrt_acp[t].unsqueeze(-1) * x0 + self.sqrt_one_minus_acp[t].unsqueeze(-1) * noise

    def diffusion_loss(self, x0):
        """Epsilon-prediction MSE with one random timestep per frame."""
        t = torch.randint(0, self.num_steps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        eps_hat = self.forward(self.q_sample(x0, t, noise), t)
        return nn.functional.mse_loss(eps_hat, noise)

    def denoise(self, x, t_star=None):
        """SDEdit refinement: noise x to level t*, then DDIM (eta=0) back to t=0.

        Deterministic given x: the forward-noise draw uses a generator seeded with
        `sdedit_seed`, independent of global RNG state.

        Differentiable: gradients from the task loss flow back through the t*+1 reverse
        steps into diff_net and on into the fusion weights.
        """
        t_star = self.sdedit_t if t_star is None else int(max(0, min(t_star, self.num_steps - 1)))

        gen = torch.Generator(device=x.device)
        gen.manual_seed(self.sdedit_seed)
        noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)

        t = torch.full((x.size(0),), t_star, device=x.device, dtype=torch.long)
        x_t = self.q_sample(x, t, noise)

        for i in reversed(range(t_star + 1)):
            t_i = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
            eps = self.forward(x_t, t_i)
            x0 = (x_t - self.sqrt_one_minus_acp[i] * eps) / self.sqrt_acp[i]
            acp_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.ones((), device=x.device)
            x_t = acp_prev.sqrt() * x0 + (1.0 - acp_prev).sqrt() * eps
        return x_t
