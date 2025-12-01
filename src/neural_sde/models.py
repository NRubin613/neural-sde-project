import torch
import torch.nn as nn


class TimeDistributedMLP(nn.Module):
    """Apply the same MLP at every time step of a sequence.

    Expects input of shape ``(batch, time, in_dim)`` and returns
    ``(batch, time, out_dim)``.
    """

    def __init__(self, in_dim = 5, hidden = 64, out_dim = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, D) got shape {x.shape}")
        b, t, d = x.shape
        x_flat = x.reshape(b * t, d)
        out = self.net(x_flat)
        return out.reshape(b, t, -1)


class DriftNet(nn.Module):
    """Neural network parameterising the drift μ(features).

    We bound the output with ``tanh`` to avoid numerical blow‑ups when
    simulating.  The scale can be tuned via ``drift_scale``.
    """

    def __init__(self, in_dim=5, hidden = 64, drift_scale = 1):
        super().__init__()
        self.tmlp = TimeDistributedMLP(in_dim= in_dim, hidden=hidden, out_dim=1)
        self.drift_scale = drift_scale

        # Initialise final layer near zero to start with small drifts
        final_layer = self.tmlp.net[-1]
        nn.init.uniform_(final_layer.weight, -1e-4, 1e-4) # Near zero
        nn.init.zeros_(final_layer.bias)

    def forward(self, x_t) -> torch.Tensor:
        mu_raw = self.tmlp(x_t)
        #return mu_raw
        return self.drift_scale * torch.tanh(mu_raw)


class DiffusionNet(nn.Module):
    """Neural network parameterising the diffusion σ(features) > 0."""

    def __init__(self, in_dim = 5, hidden = 64, volatility_max = 1.0):
        super().__init__()
        self.tmlp = TimeDistributedMLP(in_dim=in_dim, hidden=hidden, out_dim=1)
        self.volatility_max = volatility_max

    def forward(self, x_t) -> torch.Tensor:
        sigma_raw = self.tmlp(x_t)
        # softplus keeps σ strictly positive but behaves like identity for
        # large |x|; the small epsilon protects against log(0) in the loss.
        sigma = torch.nn.functional.softplus(sigma_raw) + 1e-6
        sigma = torch.clamp(sigma, max = self.volatility_max)  # prevent extreme values
        return sigma


class NeuralSDE(nn.Module):
    """Wrapper exposing ``μ(x, t)`` and ``σ(x, t)`` as a single module."""

    def __init__(self, in_dim=5, hidden = 64, drift_scale = 1, volatility_max = 2.0):
        super().__init__()
        self.mu = DriftNet(in_dim=in_dim, hidden=hidden, drift_scale=drift_scale)
        self.sigma = DiffusionNet(in_dim= in_dim, hidden=hidden, volatility_max=volatility_max)

    def forward(self, x_t):
        return self.mu(x_t), self.sigma(x_t)
