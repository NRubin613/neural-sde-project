import torch
import torch.nn as nn


class TimeDistributedMLP(nn.Module):
    """Apply the same MLP at every time step of a sequence.

    Expects input of shape ``(batch, time, in_dim)`` and returns
    ``(batch, time, out_dim)``.
    """

    def __init__(self, in_dim: int = 2, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, D) got shape {x.shape}")
        b, t, d = x.shape
        x_flat = x.reshape(b * t, d)
        out = self.net(x_flat)
        return out.reshape(b, t, -1)


class DriftNet(nn.Module):
    """Neural network parameterising the drift μ(x, t).

    We bound the output with ``tanh`` to avoid numerical blow‑ups when
    simulating.  The scale can be tuned via ``drift_scale``.
    """

    def __init__(self, hidden: int = 64, drift_scale: float = 0.1):
        super().__init__()
        self.tmlp = TimeDistributedMLP(in_dim=2, hidden=hidden, out_dim=1)
        self.drift_scale = drift_scale

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_t, t], dim=-1)
        mu_raw = self.tmlp(inp)
        return self.drift_scale * torch.tanh(mu_raw)


class DiffusionNet(nn.Module):
    """Neural network parameterising the diffusion σ(x, t) > 0."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.tmlp = TimeDistributedMLP(in_dim=2, hidden=hidden, out_dim=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_t, t], dim=-1)
        sigma_raw = self.tmlp(inp)
        # softplus keeps σ strictly positive but behaves like identity for
        # large |x|; the small epsilon protects against log(0) in the loss.
        return torch.nn.functional.softplus(sigma_raw) + 1e-6


class NeuralSDE(nn.Module):
    """Wrapper exposing ``μ(x, t)`` and ``σ(x, t)`` as a single module."""

    def __init__(self, hidden: int = 64, drift_scale: float = 0.1):
        super().__init__()
        self.mu = DriftNet(hidden=hidden, drift_scale=drift_scale)
        self.sigma = DiffusionNet(hidden=hidden)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        return self.mu(x_t, t), self.sigma(x_t, t)
