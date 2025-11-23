#!/usr/bin/env python3
import argparse
import json
import os
import matplotlib.pyplot as plt

import torch

from src.neural_sde.data import (
    load_or_download,
    to_log_prices,
    make_windows,
    TimeSeriesWindowsDataset,
)
from src.neural_sde.models import NeuralSDE
from src.neural_sde.train import train_mle


def parse_args():
    p = argparse.ArgumentParser(description="Train neural SDE on price data.")
    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--data_dir", type=str, default="data")

    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--delta", type=float, default=1.0 / 252.0)
    p.add_argument("--l2_mu", type=float, default=1)
    p.add_argument("--l2_sigma", type=float, default=1e-1)

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_dir", type=str, default="checkpoints")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[train] Loading data...")
    df = load_or_download(
        data_dir=args.data_dir,
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )

    # Log-prices in real scale
    log_prices = to_log_prices(df["close"])

    plt.plot(log_prices, label="Log-prices for groundtruth data")
    plt.xlabel("Time step")
    plt.ylabel("Log-price")
    plt.legend()
    plot_path = os.path.join(args.out_dir, "log_prices.png")
    plt.savefig(plot_path)
    plt.clf()
    print(f"[train] Saved log-price plot to {plot_path}")

    # Global normalisation
    mu = float(log_prices.mean())
    std = float(log_prices.std())
    if std <= 0:
        raise ValueError("Standard deviation of log-prices is zero; can't normalise.")
    log_norm = (log_prices - mu) / std
    
    print(f"[train] Normalisation: mu={mu:.4f}, std={std:.4f}")

    # Windows in *normalised* space; no per-window normalisation
    windows = make_windows(log_norm, seq_len=args.seq_len, normalise=False)
    dataset = TimeSeriesWindowsDataset(windows)

    # Model + training
    device = torch.device(args.device)
    model = NeuralSDE(hidden=args.hidden)
    print(f"[train] Model has {sum(p.numel() for p in model.parameters())} parameters.")

    model = train_mle(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        delta=args.delta,
        l2_mu=args.l2_mu,
        l2_sigma=args.l2_sigma,
    )

    # Save model
    model_path = os.path.join(args.out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[train] Saved model to {model_path}")

    # Save normalisation stats (used by simulate/evaluate)
    stats = {
        "mu": mu,
        "std": std,
        "last_log_price": float(log_prices[0]),
        "ticker": args.ticker,
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
    }
    stats_path = os.path.join(args.out_dir, "normalisation.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[train] Saved normalisation stats to {stats_path}")


if __name__ == "__main__":
    main()
