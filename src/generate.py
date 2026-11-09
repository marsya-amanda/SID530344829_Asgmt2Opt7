import numpy as np, torch
from foley.constants import *
from foley import audio, descriptors
from foley.model import UNet
from foley.diffusion import Diffusion

def load_ckpt(path, device):
    """Load a tensor-only checkpoint and construct model and diffusion objects."""
    ck = torch.load(path, map_location=device, weights_only=True)
    model = UNet(**ck["cfg"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Diffusion(**ck["cfg"]["diffusion"]).to(device), ck

@torch.no_grad()
def generate(model, diff, ck, class_ids, seeds, w, steps, controls=None, device="cpu"):
    """Generate seeded WAVs and return them with denormalised ``(B,80,345)`` mels."""
    if len(class_ids) != len(seeds) or not class_ids:
        raise ValueError("class_ids and seeds must be non-empty and equally sized")
    if any(not 0 <= int(c) < N_CLASSES for c in class_ids):
        raise ValueError("class ids must be in [0,6]")
    x_T = torch.stack([torch.randn(1, N_MELS, T_PAD, generator=torch.Generator().manual_seed(int(s)))
                       for s in seeds]).to(device)
    y = torch.tensor(class_ids, device=device)
    ctrl = mask = None
    if model.ctrl is not None:
        if ck.get("ctrl_stats") is None:
            raise ValueError("control checkpoint is missing ctrl_stats")
        us, ms = zip(*[descriptors.from_user(c, ck["ctrl_stats"], **(controls or {})) for c in class_ids])
        ctrl, mask = torch.tensor(np.stack(us), device=device), torch.tensor(np.stack(ms), device=device)
    elif controls and any(v is not None for v in controls.values()):
        raise ValueError("This checkpoint has no control branch; use weights/ctrl.pt")
    x = diff.ddim_sample(model, x_T, y, w, steps, ctrl, mask)
    lm = audio.denormalise(x[:, 0].cpu().numpy(), **ck["norm"])
    return [audio.logmel_to_wav(lm[i], seed=int(seeds[i])) for i in range(len(seeds))], lm
