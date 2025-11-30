#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from src.neural_sde.models import NeuralSDE
from src.neural_sde.simulate import simulate_paths


def parse_args():
    p = argparse.ArgumentParser(description="Simulate paths from trained neural SDE.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model .pt file.")
    p.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Path to normalisation.json (defaults to alongside ckpt).",
    )
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--delta", type=float, default=1.0 / 252.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--x0",
        type=float,
        default=None,
        help="Initial REAL log-price. If omitted, uses last_log_price from stats.",
    )
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--volatility_max", type=float, default=1.0)

    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Filename for the plot (saved under plots/ if relative).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # --- normalisation stats ---
    stats_path = (
        args.stats
        if args.stats is not None
        else os.path.join(os.path.dirname(args.ckpt), "normalisation.json")
    )

    with open(stats_path, "r") as f:
        stats = json.load(f)
    mu = float(stats["mu"])
    std = float(stats["std"])
    last_log_price = float(stats["last_log_price"])

    if args.x0 is None:
        x0_real = last_log_price
        print(f"[sim] Using last_log_price from stats as x0_real: {x0_real:.4f}")
    else:
        x0_real = args.x0
        print(f"[sim] Using user-provided x0_real: {x0_real:.4f}")

    # Convert initial condition to normalised space
    x0_norm = (x0_real - mu) / std

    # --- load model ---
    device = torch.device(args.device)
    model = NeuralSDE(hidden=args.hidden, volatility_max=args.volatility_max).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("[sim] Simulating paths in normalised space...")
    paths_norm = simulate_paths(
        model,
        x0=x0_norm,
        steps=args.steps,
        delta=args.delta,
        device=device,
        n=args.n,
    )

    # De-normalise to REAL log-prices
    paths_real = [[mu + std * x for x in path] for path in paths_norm]

    # --- plotting ---
    plt.figure(figsize=(8, 6))
    for i, path in enumerate(paths_real, 1):
        plt.plot(path, label=f"path {i}")
    plt.title("Simulated log-price paths")
    plt.xlabel("t")
    plt.ylabel("log-price")
    plt.legend()
    plt.tight_layout()

    # determine output path (default: plots/simulated_paths.png)
    if args.out is None:
        filename = "simulated_paths.png"
    else:
        filename = args.out

    # if it's a bare filename, put it under plots/
    if not os.path.isabs(filename) and os.path.dirname(filename) == "":
        out_path = os.path.join("plots", filename)
    else:
        out_path = filename

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[sim] Saved figure to {out_path}")


if __name__ == "__main__":
    main()
