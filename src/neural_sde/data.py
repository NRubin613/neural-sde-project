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


# Preprocessing: log prices and windows


def to_log_prices(close_series) -> np.ndarray:
    """Convert a Series/array of prices to log-prices as a 1D NumPy array."""
    arr = np.asarray(close_series, dtype=float).reshape(-1)
    if np.any(arr <= 0):
        raise ValueError("prices must be strictly positive to take logs")
    return np.log(arr)


def make_windows(x: np.ndarray, seq_len, normalise = False) -> np.ndarray:
    """
    Slice a 1D array into overlapping windows of length ``seq_len``.

    NOTE: By default we do *not* normalise per-window; we use global
    normalisation in the training script so that scale is consistent
    between training and simulation.
    """
    x = np.asarray(x, dtype=float).reshape(-1)  # ensure 1D

    if seq_len < 2 or seq_len > x.shape[0]:
        raise ValueError("seq_len must be at least 2 and at most len(x)")

    windows = []
    for start_idx in range(x.shape[0] - seq_len + 1):
        w = x[start_idx : start_idx + seq_len].astype("float32")
        if normalise:
            m = w.mean()
            s = w.std()
            if s > 0:
                w = (w - m) / s
            else:
                w = w - m
        windows.append(w)

    return np.stack(windows, axis=0)


# Dataset wrapper


class TimeSeriesWindowsDataset(Dataset):
    """Simple Dataset wrapper around a ``(N, L)`` array of windows."""

    def __init__(self, windows):
        super().__init__()
        if windows.ndim != 2:
            raise ValueError("windows should be 2D (num_windows, L)")
        # (N, L, 1)
        self.windows = torch.from_numpy(windows).float().unsqueeze(-1)

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        # returns (L, 1)
        return self.windows[idx]
