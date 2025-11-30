import torch
import numpy as np


@torch.no_grad()
def simulate_paths(
    model: torch.nn.Module,
    engine,
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

    paths = []

    # Backup engine state
    base_state = {
        'prev_price': engine.prev_price,
        'ema_var': engine.ema_var,
        'ema_trend': engine.ema_trend,
        'avg_gain': engine.avg_gain,
        'avg_loss': engine.avg_loss
    }

    for _ in range(n):
        # Reset Engine
        for k, v in base_state.items():
            setattr(engine, k, v)

        current_price = float(x0)
        path = [current_price]

        for i in range(steps):

            # Get Features from Engine
            features = engine.update_simulation(current_price).unsqueeze(0).to(device)
            
            # Predict
            mu, sigma = model(features.unsqueeze(1))
            mu = mu.item()
            sigma = sigma.item()
            #print(mu, sigma)
            
            #Geometric Brownian Motion
            noise = np.random.normal(0, 1)
            # Log-return approximation
            ret = mu * delta + sigma * noise * np.sqrt(delta)
            
            current_price = current_price * np.exp(ret)
            path.append(current_price)
            
        paths.append(path)

    return paths
