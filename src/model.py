import math, torch, torch.nn as nn, torch.nn.functional as F
from foley.constants import N_CLASSES, NULL_ID

def sinusoidal(t, dim):                              # t: (B,) float
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)

class ResBlock(nn.Module):
    def __init__(self, i, o, cd, groups=8):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(groups, i), nn.Conv2d(i, o, 3, padding=1)
        self.film = nn.Linear(cd, 2 * o)
        self.n2, self.c2 = nn.GroupNorm(groups, o), nn.Conv2d(o, o, 3, padding=1)
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(c)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + scale) + shift          # FiLM
        return self.c2(F.silu(h)) + self.skip(x)

class SelfAttn2d(nn.Module):
    def __init__(self, ch, heads=4, groups=8):
        super().__init__()
        self.n, self.qkv, self.proj, self.h = nn.GroupNorm(groups, ch), nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1), heads
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * W).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v)                   # (B,h,HW,C/h)
        return x + self.proj(o.transpose(-1, -2).reshape(B, C, H, W))

class ResAttn(nn.Module):
    def __init__(self, i, o, cd, attn):
        super().__init__(); self.res = ResBlock(i, o, cd); self.attn = SelfAttn2d(o) if attn else nn.Identity()
    def forward(self, x, c): return self.attn(self.res(x, c))

class Down(nn.Module):
    def __init__(self, ch): super().__init__(); self.c = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.c(x)
class Up(nn.Module):
    def __init__(self, ch): super().__init__(); self.c = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x): return self.c(F.interpolate(x, scale_factor=2, mode="nearest"))

class ControlEmbedding(nn.Module):
    """[Bonus] u:(B,3) in [0,1] = (brightness, onset, density); mask:(B,3) 1=specified, 0=unspecified."""
    def __init__(self, dim, n=3, fdim=64):
        super().__init__(); self.fdim = fdim
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(fdim, dim), nn.SiLU(), nn.Linear(dim, dim)) for _ in range(n)])
        self.null = nn.Parameter(torch.zeros(n, dim))
        for m in self.mlps: nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)   # starts as the core model
    def forward(self, u, mask):
        out = 0
        for k, m in enumerate(self.mlps):
            mk = mask[:, k:k + 1]
            out = out + mk * m(sinusoidal(u[:, k] * 1000, self.fdim)) + (1 - mk) * self.null[k]
        return out

class UNet(nn.Module):
    def __init__(self, base=64, mults=(1, 2, 4), n_res=2, cond_dim=256, attn_levels=(2,), use_controls=False):
        super().__init__()
        if base % 8 or any(base * m % 8 for m in mults):
            raise ValueError("base*mults must be divisible by 8 for GroupNorm")
        if len(mults) < 1:
            raise ValueError("mults must not be empty")
        cd, L = cond_dim, len(mults)
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.class_emb = nn.Embedding(N_CLASSES + 1, cd)              # index 7 = NULL
        self.ctrl = ControlEmbedding(cd) if use_controls else None
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down, skips, ch = nn.ModuleList(), [base], base
        for i, m in enumerate(mults):
            for _ in range(n_res):
                self.down.append(ResAttn(ch, base * m, cd, i in attn_levels)); ch = base * m; skips.append(ch)
            if i < L - 1: self.down.append(Down(ch)); skips.append(ch)
        self.mid1, self.mid_attn, self.mid2 = ResBlock(ch, ch, cd), SelfAttn2d(ch), ResBlock(ch, ch, cd)
        self.up = nn.ModuleList()
        for i in reversed(range(L)):
            for j in range(n_res + 1):
                self.up.append(ResAttn(ch + skips.pop(), base * mults[i], cd, i in attn_levels)); ch = base * mults[i]
                if i > 0 and j == n_res: self.up.append(Up(ch))
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t, y, ctrl=None, mask=None):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"expected x=(B,1,H,W), got {tuple(x.shape)}")
        if y.ndim != 1 or y.shape[0] != x.shape[0] or y.min() < 0 or y.max() > NULL_ID:
            raise ValueError("class ids must be a batch-sized vector in [0,7]")
        c = self.time_mlp(sinusoidal(t, self.base)) + self.class_emb(y)
        if self.ctrl is not None:
            if ctrl is None or mask is None:
                raise ValueError("control checkpoint requires ctrl and mask tensors")
            c = c + self.ctrl(ctrl, mask)
        h = self.in_conv(x); hs = [h]
        for b in self.down:
            h = b(h, c) if isinstance(b, ResAttn) else b(h); hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, c)), c)
        for b in self.up:
            if isinstance(b, ResAttn):
                skip = hs.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    raise ValueError(f"U-Net skip spatial mismatch: {h.shape} vs {skip.shape}")
                h = b(torch.cat([h, skip], 1), c)
            else:
                h = b(h)
        return self.out(h)
