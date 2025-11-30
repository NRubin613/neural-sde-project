#!/usr/bin/env python3
import argparse
import pickle
import os
import matplotlib.pyplot as plt
import torch

from src.neural_sde.models import NeuralSDE
from src.neural_sde.simulate import simulate_paths


def parse_args():
    p = argparse.ArgumentParser(description="Simulate paths from trained neural SDE.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model .pt file.")
    # Removed --stats argument as we now use engine.pkl
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--delta", type=float, default=1.0 / 252.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--x0",
        type=float,
        default=None,
        help="Initial REAL price. If omitted, continues from where training data ended.",
    )
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--volatility_max", type=float, default=2.0)

    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Filename for the plot (saved under plots/ if relative).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # --- Load Engine ---
    # We expect engine.pkl to be in the same folder as the checkpoint
    engine_path = os.path.join(os.path.dirname(args.ckpt), "engine.pkl")
    
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Could not find engine.pkl at {engine_path}. Did you train with the new train_sde.py?")
        
    with open(engine_path, "rb") as f:
        engine = pickle.load(f)
        
    print(f"[sim] Loaded Feature Engine from {engine_path}")

    # --- Determine Initial Price ---
    if args.x0 is None:
        # The engine stores the last price seen during training/priming
        if engine.prev_price is None:
            raise ValueError("Engine state is empty. Cannot infer initial price.")
        x0_real = engine.prev_price
        print(f"[sim] Using engine's last training price as start: ${x0_real:.2f}")
    else:
        x0_real = args.x0
        print(f"[sim] Using user-provided start price: ${x0_real:.2f}")

    # --- Load Model ---
    device = torch.device(args.device)
    # Note: in_dim defaults to 5 in the new model definition
    model = NeuralSDE(hidden=args.hidden, volatility_max=args.volatility_max).to(device)
    
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[sim] Simulating {args.n} paths for {args.steps} steps...")
    
    # --- Simulate ---
    # The new simulate_paths returns REAL prices directly
    paths_real = simulate_paths(
        model,
        engine=engine,
        x0=x0_real,
        steps=args.steps,
        delta=args.delta,
        device=device,
        n=args.n,
    )

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    for i, path in enumerate(paths_real, 1):
        plt.plot(path, label=f"Path {i}", alpha=0.7)
        
    plt.title("Simulated Price Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Price ($)")
    plt.grid(True, alpha=0.3)
    
    # Only show legend if few paths
    if args.n <= 10:
        plt.legend()
    
    plt.tight_layout()

    # Output handling
    if args.out is None:
        filename = "simulated_paths.png"
    else:
        filename = args.out

    if not os.path.isabs(filename) and os.path.dirname(filename) == "":
        out_path = os.path.join("plots", filename)
    else:
        out_path = filename

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[sim] Saved figure to {out_path}")


if __name__ == "__main__":
    main()