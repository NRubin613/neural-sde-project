import argparse
import pickle
import os
import matplotlib.pyplot as plt
import torch

from src.neural_sde.data import load_or_download, TimeSeriesWindowsDataset
from src.neural_sde.models import NeuralSDE
from src.neural_sde.train import train_mle
from src.neural_sde.engine import UniversalFeatureEngine

# Helper to slice windows from Feature tensors
def make_feature_windows(features, targets, seq_len):
    num_samples = features.shape[0]
    w_feats, w_targs = [], []
    for i in range(num_samples - seq_len + 1):
        w_feats.append(features[i : i + seq_len])
        w_targs.append(targets[i : i + seq_len])
    return torch.stack(w_feats), torch.stack(w_targs)

def parse_args():
    p = argparse.ArgumentParser(description="Train neural SDE on price data.")
    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Data
    print("[train] Loading data...")
    df = load_or_download(args.data_dir, args.ticker, args.start, args.end, args.interval)
    prices = df["close"].values

    # 2. Feature Engineering
    print("[train] Generating Recursive Features...")
    engine = UniversalFeatureEngine()
    
    # This generates normalized features AND the next-step return targets
    features, targets = engine.fit_transform(prices)
    
    print(f"[train] Features shape: {features.shape}, Targets shape: {targets.shape}")

    # 3. Create Dataset (Sliding Windows)
    w_feats, w_targs = make_feature_windows(features, targets, args.seq_len)
    dataset = TimeSeriesWindowsDataset(w_feats, w_targs)

    # 4. Model Setup
    device = torch.device("cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")
    
    model = NeuralSDE(in_dim=5, hidden=args.hidden)
    
    # 5. Train
    print("[train] Starting training...")
    # Note: We need to modify train_mle in src/neural_sde/train.py slightly 
    # to accept (batch_x, batch_y) from the dataset. 
    # (See file below for that update)
    model = train_mle(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    # 6. Save Model & Engine
    model_path = os.path.join(args.out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    
    engine_path = os.path.join(args.out_dir, "engine.pkl")
    with open(engine_path, "wb") as f:
        pickle.dump(engine, f)
        
    print(f"[train] Saved model to {model_path}")
    print(f"[train] Saved engine to {engine_path}")

if __name__ == "__main__":
    main()