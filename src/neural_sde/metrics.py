'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis


def acf(x, max_lag: int = 40) -> np.ndarray:
    """Plain autocorrelation function up to ``max_lag``."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2 :]
    ac /= ac[0]
    return ac[: max_lag + 1]


def compute_metrics(real_series, gen_series):
    """Compute a few simple summary statistics on returns.

    Parameters
    ----------
    real_series, gen_series:
        1‑D arrays of log‑prices (or prices).  We work with their differences.
    """
    real_series = np.asarray(real_series, dtype=float)
    gen_series = np.asarray(gen_series, dtype=float)

    real_ret = np.diff(real_series)
    gen_ret = np.diff(gen_series)

    out = {
        "real_mean": float(real_ret.mean()),
        "real_std": float(real_ret.std()),
        "real_kurtosis": float(kurtosis(real_ret, fisher=False)),
        "gen_mean": float(gen_ret.mean()),
        "gen_std": float(gen_ret.std()),
        "gen_kurtosis": float(kurtosis(gen_ret, fisher=False)),
        "acf_abs_real": acf(np.abs(real_ret)).tolist(),
        "acf_abs_gen": acf(np.abs(gen_ret)).tolist(),
    }
    return out


def plot_comparison(real_series, gen_series, save_path=None, title: str = "Real vs generated log‑prices"):
    """Plot time series, return histograms and |return| ACF side by side."""
    real_series = np.asarray(real_series, dtype=float)
    gen_series = np.asarray(gen_series, dtype=float)
    real_ret = np.diff(real_series)
    gen_ret = np.diff(gen_series)

    # 1) time series
    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(real_series, label="real", alpha=0.8)
    plt.plot(gen_series, label="generated", alpha=0.8)
    plt.xlabel("t")
    plt.ylabel("log‑price")
    plt.title(title)
    plt.legend()

    # 2) histograms of returns
    fig2 = plt.figure(figsize=(8, 4))
    bins = 50
    plt.hist(real_ret, bins=bins, density=True, alpha=0.6, label="real")
    plt.hist(gen_ret, bins=bins, density=True, alpha=0.6, label="generated")
    plt.xlabel("return")
    plt.ylabel("density")
    plt.title("Return distribution")
    plt.legend()

    # 3) ACF of absolute returns
    lags = np.arange(41)
    fig3 = plt.figure(figsize=(8, 4))
    plt.stem(lags, acf(np.abs(real_ret)), label="real |r| ACF")
    plt.stem(lags + 0.2, acf(np.abs(gen_ret)), label="generated |r| ACF")
    plt.xlabel("lag")
    plt.ylabel("correlation")
    plt.title("ACF(|returns|)")
    plt.legend()

    if save_path is not None:
        fig1.savefig(save_path.replace(".png", "_series.png"), bbox_inches="tight")
        fig2.savefig(save_path.replace(".png", "_hist.png"), bbox_inches="tight")
        fig3.savefig(save_path.replace(".png", "_acf.png"), bbox_inches="tight")

    return fig1, fig2, fig3
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import os


def compute_metrics(real_log, gen_log):
    """
    real_log, gen_log: 1D sequences of log-prices
    """
    real_log = np.asarray(real_log, dtype=float)
    gen_log  = np.asarray(gen_log, dtype=float)

    real_ret = np.diff(real_log)
    gen_ret  = np.diff(gen_log)

    metrics = {}

    # basic moments of returns
    for name, r in [("real", real_ret), ("gen", gen_ret)]:
        metrics[f"{name}_mean"]     = float(np.mean(r))
        metrics[f"{name}_std"]      = float(np.std(r, ddof=1))
        metrics[f"{name}_kurtosis"] = float(kurtosis(r, fisher=False))

    return metrics


def _acf(x, nlags=40):
    """
    Simple autocorrelation function for a 1D array x.
    Returns acf[0..nlags], with acf[0] = 1.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return np.ones(1)

    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2:]      # lags 0,1,2,...
    corr = corr[1:nlags + 2]
    return corr / corr[1]


def plot_comparison(real_log, gen_log, save_path,
                    title="Real vs generated (log-price)", nlags=40):
    """
    real_log, gen_log: sequences of log-prices.

    Saves THREE separate figures, using save_path as the base name:
      - *_paths.png       : log-price comparison
      - *_returns.png     : return histograms
      - *_acf_abs.png     : ACF(|returns|) for real & generated
    """
    real_log = np.asarray(real_log, dtype=float)
    gen_log  = np.asarray(gen_log, dtype=float)

    # base name for all plots
    base, ext = os.path.splitext(save_path)
    if ext == "":
        ext = ".png"
    paths_path   = base + "_paths"   + ext
    returns_path = base + "_returns" + ext
    acf_path     = base + "_acf_abs" + ext

    # --- series & returns ---
    t = np.arange(len(real_log))
    real_ret = np.diff(real_log)
    gen_ret  = np.diff(gen_log)

    real_abs = np.abs(real_ret)
    gen_abs  = np.abs(gen_ret)

    acf_real = _acf(real_abs, nlags=nlags)
    acf_gen  = _acf(gen_abs,  nlags=nlags)
    lags = np.arange(len(acf_real))

    # 1) log-price paths
    plt.figure(figsize=(10, 4))
    plt.plot(t, real_log, label="real")
    plt.plot(t, gen_log, label="generated")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("log-price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths_path)
    plt.close()

    # 2) return histograms
    plt.figure(figsize=(10, 4))
    bins = 50
    plt.hist(real_ret, bins=bins, density=True, alpha=0.5, label="real returns")
    plt.hist(gen_ret,  bins=bins, density=True, alpha=0.5, label="generated returns")
    plt.title("Return distribution")
    plt.xlabel("log-return")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(returns_path)
    plt.close()

    # 3) ACF(|returns|)
    plt.figure(figsize=(10, 4))
    # real
    plt.stem(lags, acf_real, linefmt="C0-", markerfmt="C0o", basefmt=" ")
    # generated (slight horizontal shift so stems don't overlap)
    plt.stem(lags + 0.1, acf_gen, linefmt="C1-", markerfmt="C1o", basefmt=" ")

    plt.title("ACF(|returns|)")
    plt.xlabel("lag")
    plt.ylabel("correlation")
    plt.legend(["real |r| ACF", "generated |r| ACF"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(acf_path)
    plt.close()

    print("Saved comparison plots to:")
    print("  ", paths_path)
    print("  ", returns_path)
    print("  ", acf_path)
