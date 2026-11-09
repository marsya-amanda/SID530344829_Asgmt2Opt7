import copy, torch
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay, self.shadow = decay, copy.deepcopy(model).eval().requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.shadow.parameters(), model.parameters()): e.lerp_(p, 1 - self.decay)
        for e, b in zip(self.shadow.buffers(), model.buffers()): e.copy_(b)
