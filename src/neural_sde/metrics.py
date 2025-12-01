import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, jarque_bera
import os

def max_drawdown(prices):
    """Calculates the maximum percentage drop from a peak."""
    prices = np.array(prices)
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return drawdown.min()

def compute_metrics(real_log, gen_log):
    """
    Computes financial metrics on LOG-PRICES and RETURNS.
    """
    # Convert log-prices to Prices for Drawdown calculation
    real_prices = np.exp(real_log)
    gen_prices = np.exp(gen_log)
    
    # Log-Returns (approx % change)
    real_ret = np.diff(real_log)
    gen_ret  = np.diff(gen_log)

    metrics = {}

    for name, r, p in [("real", real_ret, real_prices), ("gen", gen_ret, gen_prices)]:
        # 1. Moments
        metrics[f"{name}_mean"]     = float(np.mean(r))
        metrics[f"{name}_std"]      = float(np.std(r, ddof=1))
        metrics[f"{name}_skew"]     = float(skew(r))
        metrics[f"{name}_kurtosis"] = float(kurtosis(r, fisher=False)) # Normal = 3.0
        
        # 2. Normality Test (Jarque-Bera)
        # Returns (statistic, p-value). We just want the statistic (magnitude of non-normality).
        jb_stat, _ = jarque_bera(r)
        metrics[f"{name}_jb"] = float(jb_stat)

        # 3. Volatility Clustering (ACF of Squared Returns at Lag 1)
        # This measures "memory" in volatility
        r_sq = r ** 2
        r_sq_mean = r_sq.mean()
        # Simple lag-1 autocorrelation of r^2
        num = np.mean((r_sq[:-1] - r_sq_mean) * (r_sq[1:] - r_sq_mean))
        denom = np.var(r_sq)
        metrics[f"{name}_vol_clust"] = float(num / (denom + 1e-9))

        # 4. Max Drawdown (Risk)
        metrics[f"{name}_mdd"] = float(max_drawdown(p))

    return metrics


def save_metrics_table(metrics, save_path):
    """
    Saves a comparison table of ADVANCED metrics as an image.
    """
    # Define Rows and Keys to extract
    table_map = [
        ("Daily Mean", "mean", "{:.6f}"),
        ("Daily Volatility", "std", "{:.4f}"),
        ("Skewness", "skew", "{:.4f}"),
        ("Kurtosis (Fat Tails)", "kurtosis", "{:.2f}"),
        ("Jarque-Bera Score", "jb", "{:.0f}"),
        ("Vol Clustering (Lag-1)", "vol_clust", "{:.4f}"),
        ("Max Drawdown", "mdd", "{:.2%}"),
    ]
    
    rows = [item[0] for item in table_map]
    cols = ["Real Data", "Neural SDE"]
    
    cell_text = []
    for label, key_suffix, fmt in table_map:
        val_real = metrics[f"real_{key_suffix}"]
        val_gen  = metrics[f"gen_{key_suffix}"]
        cell_text.append([fmt.format(val_real), fmt.format(val_gen)])

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4)) # Taller for more rows
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=cell_text,
                     rowLabels=rows,
                     colLabels=cols,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Save
    base, ext = os.path.splitext(save_path)
    if not ext: ext = ".png"
    table_path = base + "_table" + ext
    
    plt.savefig(table_path, bbox_inches='tight', dpi=150)
    plt.close()
    print("  ", table_path)

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

    # --- Plot 4: Tables ---
    metrics = compute_metrics(real_log, gen_log)
    save_metrics_table(metrics, save_path)
