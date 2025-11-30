import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Local CSV + synthetic fallback

def _make_synthetic_series(n = 2000, seed = 0) -> pd.DataFrame:
    """
    Fallback when market data is not available.

    We simulate a simple geometric Brownian motion so that the rest of the
    pipeline can still be exercised. This keeps the project runnable even
    without existing data.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu = 0.05
    sigma = 0.2

    shocks = rng.normal(
        loc=(mu - 0.5 * sigma**2) * dt,
        scale=sigma * np.sqrt(dt),
        size=n,
    )
    log_price = shocks.cumsum()
    price = np.exp(log_price)

    dates = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame({"date": dates, "close": price})


def _clean_price_csv(df) -> pd.DataFrame:
    """
    Clean a raw DataFrame that should contain at least a date column and a
    price column. 

    Robust to junk rows like ',^GSPC' in the close column and
    slightly different column names.
    """
    df = df.copy()

    # Find date column
    date_col = None
    for cand in ["date", "Date", "DATE", "timestamp", "Timestamp"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"Could not find a date column in CSV. Columns: {df.columns}")

    # Find price column
    price_col = None
    for cand in ["close", "Close", "Adj Close", "adjclose", "price", "Price"]:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        raise ValueError(f"Could not find a close/price column in CSV. Columns: {df.columns}")

    out = df[[date_col, price_col]].copy()
    out = out.rename(columns={date_col: "date", price_col: "close"})

    # Parse types robustly
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    # Drop junk rows (like ',^GSPC')
    out = out.dropna(subset=["date", "close"])

    if out.empty:
        raise ValueError("No valid rows left after cleaning CSV.")

    return out


def load_or_download(
    data_dir,
    ticker,
    start,
    end,
    interval = "1d",
) -> pd.DataFrame:
    """
    Load price data for a given ticker.

    Behaviour:
      - Look in ``data_dir`` for any CSV whose filename starts with the
        ticker.
      - If found, load the first match and clean it.
      - If not, create a synthetic GBM series, save it under a sensible
        name and return it.

    """
    os.makedirs(data_dir, exist_ok=True)

    safe_ticker = ticker.replace("^", "").replace("/", "-")

    # Any CSV that starts with this ticker is acceptable
    candidates = [
        f
        for f in os.listdir(data_dir)
        if f.startswith(safe_ticker) and f.lower().endswith(".csv")
    ]

    if candidates:
        # Take the first (or you could sort and take latest)
        path = os.path.join(data_dir, sorted(candidates)[0])
        raw = pd.read_csv(path)
        df = _clean_price_csv(raw)
        return df

    # No CSV present: create synthetic series and save it
    fname = f"{safe_ticker}_{start}_{end}_{interval}.csv"
    path = os.path.join(data_dir, fname)
    print("[data] No local CSV found; using synthetic GBM series instead.")
    df = _make_synthetic_series()
    df.to_csv(path, index=False)
    return df

# Dataset wrapper


class TimeSeriesWindowsDataset(Dataset):
    def __init__(self, features, targets):
        super().__init__()
        # features: (N, L, D)
        # targets: (N, L, 1)
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
