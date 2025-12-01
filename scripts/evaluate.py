#!/usr/bin/env python3
import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.neural_sde.data import load_or_download
from src.neural_sde.metrics import compute_metrics, plot_comparison
from src.neural_sde.models import NeuralSDE
from src.neural_sde.simulate import simulate_paths

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate neural SDE vs real data.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model .pt file.")
    # Removed --stats, added --data params to ensure we have history to prime the engine
    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--data_dir", type=str, default="data")

    p.add_argument("--steps", type=int, default=10000, help="Max steps to evaluate (clipped by data len)")
    p.add_argument("--delta", type=float, default=1.0 / 252.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--volatility_max", type=float, default=2.0)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Filename for plots (saved under plots/ if relative).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # --- Load Engine ---
    engine_path = os.path.join(os.path.dirname(args.ckpt), "engine.pkl")
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Could not find engine.pkl at {engine_path}")
        
    with open(engine_path, "rb") as f:
        engine = pickle.load(f)
    print(f"[eval] Loaded Feature Engine from {engine_path}")

    # --- Load Real Data ---
    df = load_or_download(
        data_dir=args.data_dir,
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )
    real_prices = df["close"].values

    # --- Priming Strategy ---
    # We need ~50 steps to prime the engine. 
    # We will use the first 50 steps of the loaded data for priming, 
    # and evaluate on the rest.
    warmup = 50
    if len(real_prices) < (warmup + 10):
        raise ValueError("Not enough data points for warmup + evaluation.")

    print(f"[eval] Using first {warmup} steps to prime the engine...")
    priming_data = real_prices[:warmup]
    engine.prime_engine(priming_data)

    # The simulation starts from the last price of the warmup
    start_price = real_prices[warmup - 1]
    
    # Determine evaluation slice
    eval_len = min(args.steps, len(real_prices) - warmup)
    real_eval_slice = real_prices[warmup : warmup + eval_len]
    
    print(f"[eval] Evaluating on next {eval_len} steps starting from ${start_price:.2f}")

    # --- Load Model ---
    device = torch.device(args.device)
    model = NeuralSDE(hidden=args.hidden, volatility_max=args.volatility_max).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Simulate One Path ---
    # simulate_paths returns REAL prices
    paths = simulate_paths(
        model,
        engine=engine,
        x0=start_price,
        steps=eval_len - 1, # -1 because start_price is included
        delta=args.delta,
        device=device,
        n=1,
    )
    gen_prices = np.array(paths[0])

    # Ensure alignment (sometimes off by one due to indexing)
    min_len = min(len(real_eval_slice), len(gen_prices))
    real_eval_slice = real_eval_slice[:min_len]
    gen_prices = gen_prices[:min_len]

    # Convert to Log Prices for Metric Calculation
    # (Metrics are standardly calculated on log-returns)
    real_log = np.log(real_eval_slice)
    gen_log = np.log(gen_prices)

    # --- Metrics ---
    metrics = compute_metrics(real_log, gen_log)
    print("[eval] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # --- Plotting ---
    if args.out is None:
        filename = "eval_comparison.png"
    else:
        filename = args.out

    if not os.path.isabs(filename) and os.path.dirname(filename) == "":
        out_path = os.path.join("plots", filename)
    else:
        out_path = filename

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1. Standard Comparison (Path, Returns, ACF)
    plot_comparison(
        real_log,
        gen_log,
        save_path=out_path,
        title=f"Eval: {args.ticker} (Log-Prices)",
    )
    
    # 2. Extract and Plot Drift/Diffusion Trajectories
    # We re-run a simplified pass to extract mu/sigma for visualization
    drift_vals = []
    diff_vals = []
    
    # Reset engine for extraction pass
    engine.prime_engine(priming_data)
    curr = start_price
    
    with torch.no_grad():
        for _ in range(min_len):
            feats = engine.update_simulation(curr).unsqueeze(0).unsqueeze(0).to(device)
            mu, sigma = model(feats)
            drift_vals.append(mu.item())
            diff_vals.append(sigma.item())
            # For visualization, we just use the pre-calculated path prices
            if _ < min_len - 1:
                curr = gen_prices[_+1]

    # Plot Internal Dynamics
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(drift_vals, label="Predicted Drift (μ)", color="tab:blue", alpha=0.8)
    plt.axhline(0, color="black", linestyle="--", alpha=0.3)
    plt.title("Internal SDE Dynamics over Simulation")
    plt.ylabel("Drift")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(diff_vals, label="Predicted Diffusion (σ)", color="tab:orange", alpha=0.8)
    plt.ylabel("Volatility")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    dynamics_path = out_path.replace(".png", "_dynamics.png")
    plt.tight_layout()
    plt.savefig(dynamics_path)
    print(f"[eval] Saved dynamics plot to {dynamics_path}")

if __name__ == "__main__":
    main()