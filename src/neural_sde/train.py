import math

import torch
from torch.utils.data import DataLoader

from .utils import time_grid


def gaussian_nll(dx, mean, var, eps=1e-6):
    """Element-wise negative log-likelihood of a Normal distribution."""
    var = torch.clamp(var, min=eps)
    return 0.5 * (torch.log(2 * math.pi * var) + (dx - mean) ** 2 / var)


def train_mle(
    model,
    dataset,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    delta: float = 1.0 / 252.0,
    l2_mu: float = 1e-2,
    l2_sigma: float = 1e-3,
):
    """Maximum likelihood training loop.

    The discrete-time approximation assumes that

        ΔX_t ≈ N( μ(x_t, t) Δ,  σ(x_t, t)^2 Δ )

    where Δ is `delta`. We treat all (x_t, x_{t+1}) pairs inside a window as
    conditionally independent given the model parameters.

    Regularisation:
      - l2_mu    : penalises large drift outputs μ̂ (keeps drift modest).
      - l2_sigma : penalises large |log σ̂|, i.e. avoids σ̂ being tiny or huge.
    """
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)  # (B, L, 1)
            b, length, _ = batch.shape

            # time feature in [0, 1]
            t = time_grid(batch, length)

            x_t = batch[:, :-1, :]        # (B, L-1, 1)
            x_next = batch[:, 1:, :]      # (B, L-1, 1)
            mu_hat, sigma_hat = model(x_t, t[:, :-1, :])

            dx = x_next - x_t
            mean = mu_hat * delta
            var = (sigma_hat ** 2) * delta

            # core NLL
            nll = gaussian_nll(dx, mean, var).mean()

            # drift regularisation: keep μ̂ small
            reg_mu = 0.0
            if l2_mu > 0.0:
                reg_mu = (mu_hat ** 2).mean()

            # diffusion regularisation: keep log σ̂ near 0 (σ̂ ~ 1 in normalised units)
            reg_sigma = 0.0
            if l2_sigma > 0.0:
                # +eps inside log to avoid log(0)
                log_sigma = torch.log(sigma_hat + 1e-8)
                reg_sigma = (log_sigma ** 2).mean()

            loss = nll + l2_mu * reg_mu + l2_sigma * reg_sigma

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"[epoch {epoch:03d}] loss={avg:.6f}")

    return model
