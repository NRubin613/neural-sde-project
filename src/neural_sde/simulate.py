import torch


@torch.no_grad()
def simulate_paths(
    model: torch.nn.Module,
    x0,
    steps = 256,
    delta = 1.0 / 252.0,
    device = "cpu",
    n = 1,
):
    """Simulate sample paths from a trained NeuralSDE.

    Parameters
    ----------
    model:
        Trained ``NeuralSDE`` instance.
    x0:
        Initial log‑price
    steps:
        Number of Euler–Maruyama steps to take.
    delta:
        Time step used for simulation.  Should match the value used during
        training (default is one trading day in year‑fraction units).
    device:
        Device to run the simulation on.
    n:
        Number of paths to generate.

    Returns
    -------
    
        A list containing ``n`` paths, each of length ``steps + 1``.
    """
    model.eval().to(device)
    x0 = torch.as_tensor(x0, dtype=torch.float32, device=device).view(1, 1, 1)  # (1,1,1)

    paths = []
    for _ in range(n):
        x = x0.clone()
        xs = [x.item()]
        for i in range(steps):
            t = torch.tensor(
                [[[i / max(steps - 1, 1)]]],
                dtype=torch.float32,
                device=device,
            )
            mu, sigma = model(x, t)
            #print(x, mu, sigma)
            noise = torch.randn_like(mu)
            x = x + mu * delta + sigma * noise * (delta**0.5)
            xs.append(x.item())
        paths.append(xs)

    return paths
