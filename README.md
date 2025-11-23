# Neural SDE for 1D Financial Time Series

PyTorch project to fit a **1‑dimensional stochastic differential equation**
directly to a price series and then simulate sample paths.

We model log‑prices `X_t` via

> dX_t = μ_θ(X_t, t) dt + σ_φ(X_t, t) dW_t

where both the drift `μ_θ` and diffusion `σ_φ` are feed‑forward neural
networks.  The parameters are learned by maximum likelihood on discrete
increments.


## What’s in here?

- `src/neural_sde/data.py` – read or simulate data, make overlapping
  windows and wrap them in a PyTorch `Dataset`.
- `src/neural_sde/models.py` – drift and diffusion networks plus the `NeuralSDE`
  wrapper.
- `src/neural_sde/train.py` – maximum likelihood training loop using a Gaussian
  increment approximation.
- `src/neural_sde/simulate.py` – Euler–Maruyama sampler.
- `src/neural_sde/metrics.py` – a couple of basic diagnostics (kurtosis,
  ACF of |returns|, etc.).
- `scripts/` – small command‑line entry points for training, simulating and
  comparing to real data.


## Setup

```bash
pip install -r requirements.txt
```

Everything is standard (`torch`, `numpy`, `pandas`, `matplotlib`, `os`,
`scipy`).


## Training

```bash
python3 -m scripts.train_sde --ticker ^GSPC --start 2005-01-01 --end 2025-01-01
```

This will:

1. Read daily prices from chosen csv file (or fall back to a synthetic
   GBM‑style series if the download fails).
2. Convert to log‑prices.
3. Slice into overlapping windows (default length 256) and standardise each
   window.
4. Fit the neural SDE via maximum likelihood.
5. Save the weights under `checkpoints/model.pt`.


## Simulating

```bash
python3 -m scripts.simulate_paths --ckpt checkpoints/model.pt --stats checkpoints/normalisation.json

```

This produces a plot of several simulated log‑price paths under the learned
dynamics.


## Quick evaluation

```bash
python3 -m scripts.evaluate     --ckpt checkpoints/model.pt     --ticker ^GSPC     --start 2005-01-01     --end 2025-01-01     --interval 1d
```

This loads some market data, simulates one path of the same length and plots:

- the real vs generated log‑price series,
- histograms of returns, and
- the ACF of absolute returns.

It also prints a handful of summary statistics
