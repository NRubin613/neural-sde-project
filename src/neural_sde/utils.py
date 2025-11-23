import torch

def as_tensor(x, device=None, dtype=torch.float32):
    t = torch.as_tensor(x, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t

def time_grid(batch, T):
    # normalized time 0..1 shaped (B, T, 1)
    t = torch.linspace(0.0, 1.0, steps=T, device=batch.device).view(1, T, 1)
    t = t.expand(batch.size(0), -1, -1)
    return t
