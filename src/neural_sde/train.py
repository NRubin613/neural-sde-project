import math
import torch
from torch.utils.data import DataLoader

def gaussian_nll(dx, mean, var, eps=1e-6):
    """Element-wise negative log-likelihood of a Normal distribution."""
    var = torch.clamp(var, min=eps)
    return 0.5 * (torch.log(2 * math.pi * var) + (dx - mean) ** 2 / var)


def train_mle(
    model,
    dataset,
    epochs = 20,
    batch_size = 128,
    lr = 1e-3,
    device = "cpu",
    delta = 1.0 / 252.0,
    l2_mu = 1,
    l2_sigma = 1e-3,
):
    """Maximum likelihood training loop.

    The discrete-time approximation assumes that

        ΔX_t ≈ N( μ(features) Δ,  σ(features)^2 Δ )

    where Δ is `delta`.
    """
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device) #inputs
            batch_y = batch_y.to(device) #targets

    
            mu_hat, sigma_hat = model(batch_x) # Predictions

            mean = mu_hat * delta
            var = (sigma_hat ** 2) * delta

            # core NLL
            nll = gaussian_nll(batch_y, mean, var).mean()

            # drift regularisation: keep μ small
            reg_mu = (mu_hat ** 2).mean() if l2_mu > 0 else 0.0

            # diffusion regularisation: keep log σ near 0 
            # +eps inside log to avoid log(0)
            log_sigma = torch.log(sigma_hat + 1e-8)
            reg_sigma = (log_sigma ** 2).mean() if l2_sigma > 0 else 0.0

            # temporal smoothness regularisation on μ
            """if mu_hat.shape[1] > 1:
                mu_diff = mu_hat[:, 1:, :] - mu_hat[:, :-1, :]
                reg_smooth = (mu_diff ** 2).mean()
            else:
                reg_smooth = 0.0"""

            loss = nll + l2_mu * reg_mu + l2_sigma * reg_sigma #+ 5 * reg_smooth

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"[epoch {epoch:03d}] loss={avg:.6f}")

    return model
