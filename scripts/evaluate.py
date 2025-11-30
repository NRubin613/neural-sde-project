#!/usr/bin/env python3
import argparse
import json
import os
import matplotlib.pyplot as plt


import torch

from src.neural_sde.data import load_or_download, to_log_prices
from src.neural_sde.metrics import compute_metrics, plot_comparison
from src.neural_sde.models import NeuralSDE
from src.neural_sde.simulate import simulate_paths


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate neural SDE vs real data.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model .pt file.")
    p.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Path to normalisation.json (defaults to alongside ckpt).",
    )

    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--data_dir", type=str, default="data")

    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--delta", type=float, default=1.0 / 252.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--volatility_max", type=float, default=1.0)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Filename for plots (saved under plots/ if relative).",
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

    # --- real data ---
    df = load_or_download(
        data_dir=args.data_dir,
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )
    real_log = to_log_prices(df["close"])

    if args.steps + 1 > len(real_log):
        raise ValueError("Not enough data points for the requested number of steps.")

    real_slice = real_log[0:(args.steps + 1)]

    # initial condition in normalised space
    x0_norm = (last_log_price - mu) / std

    # --- model ---
    device = torch.device(args.device)
    model = NeuralSDE(hidden=args.hidden, volatility_max=args.volatility_max).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # simulate one path
    paths_norm = simulate_paths(
        model,
        x0=x0_norm,
        steps=args.steps,
        delta=args.delta,
        device=device,
        n=1,
    )
    gen_norm = paths_norm[0]
    gen_real = [mu + std * x for x in gen_norm]

    # align lengths
    if len(gen_real) != len(real_slice):
        m = min(len(gen_real), len(real_slice))
        real_slice = real_slice[-m:]
        gen_real = gen_real[-m:]

    # metrics
    metrics = compute_metrics(real_slice, gen_real)
    print("[eval] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # --- plotting ---
    if args.out is None:
        filename = "eval_comparison.png"
    else:
        filename = args.out

    # if it's a bare filename, put it under plots/
    if not os.path.isabs(filename) and os.path.dirname(filename) == "":
        out_path = os.path.join("plots", filename)
    else:
        out_path = filename

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ##Plotting learnt mu and sigma against x
    x = torch.linspace(-3, 3, 100).view(-1, 1, 1).to(device)
    t = torch.zeros_like(x).to(device)
    with torch.no_grad():
        mu, sigma = model(x, t)

    x_vals = x.cpu().squeeze()
    mu_vals = mu.cpu().squeeze()
    sigma_vals = sigma.cpu().squeeze()

    # --- Plot 1: mu(x) ---
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, mu_vals, label="μ(x)", color="tab:blue")
    plt.title("Learnt Drift Function μ(x)")
    plt.xlabel("x (normalised log-price)")
    plt.ylabel("μ(x)")
    plt.grid(True)
    plt.tight_layout()

    mu_plot_path = out_path.replace("eval_comparison.png", "learnt_mu.png")
    plt.savefig(mu_plot_path)
    plt.close()

    # --- Plot 2: sigma(x) ---
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, sigma_vals, label="σ(x)", color="tab:orange")
    plt.title("Learnt Diffusion Function σ(x)")
    plt.xlabel("x (normalised log-price)")
    plt.ylabel("σ(x)")
    plt.grid(True)
    plt.tight_layout()

    sigma_plot_path = out_path.replace("eval_comparison.png", "learnt_sigma.png")
    plt.savefig(sigma_plot_path)
    plt.close()

    print("Saved:", mu_plot_path, "and", sigma_plot_path)

    plot_comparison(
        real_slice,
        gen_real,
        save_path=out_path,
        title="Real vs generated (log-price)",
    )
    print(f"[eval] Saved comparison plot(s) to {out_path}")


if __name__ == "__main__":
    main()
