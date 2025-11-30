import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import os

def compute_metrics(real_log, gen_log):
    """
    Computes statistical metrics on LOG-RETURNS.
    
    Parameters
    ----------
    real_log, gen_log: 1D arrays of log-prices.
    """
    real_log = np.asarray(real_log, dtype=float)
    gen_log  = np.asarray(gen_log, dtype=float)

    # Convert log-prices to log-returns (diff)
    real_ret = np.diff(real_log)
    gen_ret  = np.diff(gen_log)

    metrics = {}

    # Basic moments of returns
    for name, r in [("real", real_ret), ("gen", gen_ret)]:
        metrics[f"{name}_mean"]     = float(np.mean(r))
        metrics[f"{name}_std"]      = float(np.std(r, ddof=1))
        metrics[f"{name}_kurtosis"] = float(kurtosis(r, fisher=False)) # Fisher=False means Pearson (normal=3.0)

    return metrics


def _acf(x, nlags=40):
    """
    Autocorrelation function.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return np.ones(nlags + 1)

    # Compute autocovariance
    # Full correlation is safer than dot product loop for speed/stability
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2:]  # Keep non-negative lags
    
    # Normalize by variance (lag 0)
    if corr[0] == 0:
        return np.zeros(min(len(corr), nlags+1))
        
    corr = corr / corr[0]
    
    # Return requested lags
    return corr[:nlags + 1]


def plot_comparison(real_log, gen_log, save_path,
                    title="Real vs generated (log-price)", nlags=50):
    """
    Generates comparison plots for Log-Prices, Returns, and Volatility Clustering (ACF of absolute returns).
    """
    real_log = np.asarray(real_log, dtype=float)
    gen_log  = np.asarray(gen_log, dtype=float)

    # Base name handling
    base, ext = os.path.splitext(save_path)
    if not ext: ext = ".png"
    
    paths_path   = base + "_paths"   + ext
    returns_path = base + "_returns" + ext
    acf_path     = base + "_acf_abs" + ext

    # Calculate Returns
    t = np.arange(len(real_log))
    real_ret = np.diff(real_log)
    gen_ret  = np.diff(gen_log)

    # Calculate Absolute Returns (Proxy for Volatility)
    real_abs = np.abs(real_ret)
    gen_abs  = np.abs(gen_ret)

    # Calculate ACF of Volatility
    acf_real = _acf(real_abs, nlags=nlags)
    acf_gen  = _acf(gen_abs,  nlags=nlags)
    lags = np.arange(len(acf_real))

    # --- Plot 1: Log-Price Paths ---
    plt.figure(figsize=(10, 5))
    plt.plot(t, real_log, label="Real Data", color='black', alpha=0.7, linewidth=1)
    plt.plot(t, gen_log, label="Simulated (SDE)", color='tab:blue', alpha=0.8, linewidth=1)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Log Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths_path)
    plt.close()

    # --- Plot 2: Return Histograms (Tail Risk) ---
    plt.figure(figsize=(10, 5))
    bins = 60
    # Determine common range for histograms
    combined = np.concatenate([real_ret, gen_ret])
    r_min, r_max = np.percentile(combined, [0.5, 99.5]) # clip extreme outliers for cleaner plot
    
    plt.hist(real_ret, bins=bins, range=(r_min, r_max), density=True, alpha=0.5, color='black', label="Real Returns")
    plt.hist(gen_ret,  bins=bins, range=(r_min, r_max), density=True, alpha=0.5, color='tab:blue', label="Simulated Returns")
    plt.title("Return Distribution (Log-Scale Y for Tails)")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.yscale('log') # Log scale helps see fat tails
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(returns_path)
    plt.close()

    # --- Plot 3: Volatility Clustering (ACF of |r|) ---
    plt.figure(figsize=(10, 5))
    # Slight offset for visibility
    plt.stem(lags, acf_real, linefmt="k-", markerfmt="ko", basefmt=" ", label="Real Volatility Correlation")
    plt.stem(lags + 0.2, acf_gen, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="Simulated Volatility Correlation")
    
    plt.title("Volatility Clustering: ACF of |Returns|")
    plt.xlabel("Lag (Days)")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(acf_path)
    plt.close()

    print("Saved comparison plots to:")
    print("  ", paths_path)
    print("  ", returns_path)
    print("  ", acf_path)