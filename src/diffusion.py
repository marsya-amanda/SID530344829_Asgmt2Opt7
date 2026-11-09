import torch, torch.nn.functional as F
from foley.constants import NULL_ID

class Diffusion:
    """Linear-schedule epsilon diffusion with deterministic DDIM sampling."""
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        if T < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("invalid diffusion schedule")
        self.T = T
        self.ab = torch.cumprod(1 - torch.linspace(beta_start, beta_end, T), 0)      # \bar{alpha}_t
    def to(self, dev): self.ab = self.ab.to(dev); return self

    def q_sample(self, x0, t, eps):
        """Add noise to ``x0=(B,C,H,W)`` at integer timesteps ``t=(B,)``."""
        a = self.ab[t].view(-1, 1, 1, 1); return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def loss(self, model, x0, y, ctrl=None, mask=None, generator=None):
        """Return epsilon MSE for a batch of normalised mel tensors."""
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device, generator=generator)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        return F.mse_loss(model(self.q_sample(x0, t, eps), t, y, ctrl, mask).float(), eps)

    @torch.no_grad()
    def ddim_sample(self, model, x_T, y, w=3.0, steps=50, ctrl=None, mask=None):
        """Sample ``x_T`` with eta-zero DDIM and conditional-free guidance."""
        if steps < 1 or w < 0:
            raise ValueError("steps must be positive and guidance must be non-negative")
        ts = torch.linspace(self.T - 1, 0, steps, device=x_T.device).round().long()
        x = x_T
        for i, t in enumerate(ts):
            tt = t.expand(x.size(0))
            if w == 1.0:
                eps = model(x, tt, y, ctrl, mask)
            else:
                yn = torch.full_like(y, NULL_ID)
                c2 = None if ctrl is None else torch.cat([ctrl, torch.zeros_like(ctrl)])
                m2 = None if mask is None else torch.cat([mask, torch.zeros_like(mask)])
                e_c, e_u = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, yn]), c2, m2).chunk(2)
                eps = e_u + w * (e_c - e_u)                                  # CFG
            ab_t = self.ab[t]
            ab_prev = self.ab[ts[i + 1]] if i + 1 < len(ts) else x.new_tensor(1.0)
            x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x
