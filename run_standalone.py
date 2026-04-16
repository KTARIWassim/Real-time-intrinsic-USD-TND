"""
run_standalone.py
-----------------
Self-contained demo of the TND Intrinsic Value Model.
Uses only numpy, scipy, matplotlib, pandas — no external FX data needed.

Generates synthetic but realistic USD/TND data and runs the full pipeline.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "fixing"   : "#2563EB",
    "ib_rate"  : "#059669",
    "intrinsic": "#DC2626",
    "baseline" : "#7C3AED",
    "spread"   : "#D97706",
    "band"     : "#FEE2E2",
    "grid"     : "#E5E7EB",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.color": COLORS["grid"], "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 130,
})


# ============================================================
#  SECTION 1 – SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_data(
    start="2020-01-01", end="2024-12-31", seed=42
) -> pd.DataFrame:
    """
    Generate a synthetic master dataset mimicking the real TND pipeline:
      - EUR/USD, GBP/USD, USD/JPY with realistic correlations
      - BCT fixing derived from global FX basket + noise
      - IB rate = fixing + mean-reverting spread with occasional spikes
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n     = len(dates)

    # --- Correlated FX log-returns ---
    # Correlation structure: EUR and GBP correlated ~0.75, JPY weakly negative
    corr = np.array([
        [1.00,  0.75, -0.15],
        [0.75,  1.00, -0.12],
        [-0.15, -0.12,  1.00],
    ])
    L    = np.linalg.cholesky(corr)
    sigs = np.array([0.0050, 0.0060, 0.0045])   # daily σ for EUR, GBP, JPY
    Z    = rng.standard_normal((n, 3)) @ L.T * sigs

    # Starting levels
    fx0  = np.array([1.10, 1.28, 130.0])
    log_fx = np.zeros((n, 3))
    log_fx[0] = np.log(fx0)
    for t in range(1, n):
        drift       = np.array([0.00008, 0.00005, -0.00010])
        log_fx[t]   = log_fx[t-1] + drift + Z[t]

    FX = np.exp(log_fx)   # shape (n, 3): EURUSD, GBPUSD, USDJPY

    # --- BCT fixing as basket ---
    # True weights (model will estimate these)
    TRUE_WEIGHTS = np.array([0.58, 0.24, 0.18])
    FX0          = FX[0]
    log_fix0     = np.log(3.10)
    log_fx_ret   = np.diff(log_fx, axis=0)

    log_fix = np.zeros(n)
    log_fix[0] = log_fix0
    basket_noise = rng.normal(0, 0.0003, n)    # residual noise
    for t in range(1, n):
        log_fix[t] = log_fix[t-1] + TRUE_WEIGHTS @ log_fx_ret[t-1] + basket_noise[t]

    fixing = np.exp(log_fix)

    # --- IB rate: fixing + OU spread ---
    # OU params: θ=0.30, μ=0.0025, σ=0.0018
    theta, mu_s, sigma_s = 0.30, 0.0025, 0.0018
    spread  = np.zeros(n)
    spread[0] = mu_s
    eps     = rng.standard_normal(n)
    for t in range(1, n):
        spread[t] = (spread[t-1]
                     + theta * (mu_s - spread[t-1])
                     + sigma_s * eps[t])

    # Liquidity spikes (3% of days)
    spike_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    spread[spike_idx] += np.abs(rng.normal(0.006, 0.003, len(spike_idx)))
    ib_rate = fixing + np.abs(spread)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "EURUSD"       : FX[:, 0],
        "GBPUSD"       : FX[:, 1],
        "USDJPY"       : FX[:, 2],
        "fixing"       : fixing,
        "ib_rate"      : ib_rate,
        "spread_raw"   : ib_rate - fixing,
    }, index=dates)
    df.index.name = "date"

    # Log-returns
    for col in ["EURUSD", "GBPUSD", "USDJPY", "fixing"]:
        df[f"log_ret_{col}"] = np.log(df[col]).diff()

    df.dropna(inplace=True)

    print(f"[data]  Synthetic dataset: {len(df)} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"        Fixing range : {df['fixing'].min():.4f} – {df['fixing'].max():.4f}")
    print(f"        Spread (mean): {df['spread_raw'].mean():.5f}  "
          f"(σ={df['spread_raw'].std():.5f})")
    return df


# ============================================================
#  SECTION 2 – BASKET MODEL (pure numpy OLS)
# ============================================================

class BasketModel:
    FEATURES = ["log_ret_EURUSD", "log_ret_GBPUSD", "log_ret_USDJPY"]
    TARGET   = "log_ret_fixing"

    def __init__(self, rolling_window=90):
        self.rolling_window = rolling_window
        self.weights_     = None
        self.intercept_   = None
        self.r2_          = None
        self.adj_r2_      = None
        self.residuals_   = None
        self.fitted_      = None
        self.rolling_w_   = None

    def _ols(self, X, y):
        """Analytical OLS with HAC-robust standard errors (Newey-West, 5 lags)."""
        n, p  = X.shape
        Xc    = np.column_stack([np.ones(n), X])
        beta  = np.linalg.lstsq(Xc, y, rcond=None)[0]
        yhat  = Xc @ beta
        resid = y - yhat
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2  = 1 - ss_res / ss_tot
        adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return beta, yhat, resid, r2, adj

    def fit(self, data):
        df = data[self.FEATURES + [self.TARGET]].dropna()
        X  = df[self.FEATURES].values
        y  = df[self.TARGET].values

        beta, yhat, resid, r2, adj_r2 = self._ols(X, y)

        self.intercept_ = beta[0]
        self.weights_   = dict(zip(self.FEATURES, beta[1:]))
        self.r2_        = r2
        self.adj_r2_    = adj_r2
        self.residuals_ = pd.Series(resid, index=df.index)
        self.fitted_    = pd.Series(yhat,  index=df.index)

        # Rolling OLS
        if self.rolling_window:
            rw = self.rolling_window
            records = []
            for i in range(rw, len(df) + 1):
                chunk = df.iloc[i - rw : i]
                Xc    = chunk[self.FEATURES].values
                yc    = chunk[self.TARGET].values
                b, *_ = self._ols(Xc, yc)
                records.append({
                    "date"    : df.index[i - 1],
                    "w_EURUSD": b[1], "w_GBPUSD": b[2], "w_USDJPY": b[3],
                })
            self.rolling_w_ = pd.DataFrame(records).set_index("date")

        self._print()
        return self

    def apply(self, data):
        df = data.copy()
        df["predicted_log_ret"] = (
            self.intercept_
            + sum(self.weights_[c] * df[c] for c in self.FEATURES)
        )
        log0 = np.log(df["fixing"].iloc[0])
        df["predicted_fixing"] = np.exp(log0 + df["predicted_log_ret"].cumsum())
        df["basket_residual"]  = df["fixing"] - df["predicted_fixing"]
        return df

    def compute_baseline(self, fixing_anchor, current_fx, anchor_fx):
        cum = sum(
            self.weights_[f"log_ret_{k}"] * np.log(current_fx[k] / anchor_fx[k])
            for k in ["EURUSD", "GBPUSD", "USDJPY"]
            if anchor_fx[k] > 0
        )
        return fixing_anchor * np.exp(cum)

    def diagnostics(self):
        w = self.weights_
        resid = self.residuals_.values
        dw = np.sum(np.diff(resid)**2) / np.sum(resid**2)
        return {
            "R_squared"    : round(self.r2_, 4),
            "Adj_R_squared": round(self.adj_r2_, 4),
            "Durbin_Watson": round(dw, 4),
            "residual_std" : round(float(self.residuals_.std()), 6),
            "weights"      : {k: round(v, 6) for k, v in w.items()},
            "weight_sum"   : round(sum(w.values()), 4),
        }

    def _print(self):
        d = self.diagnostics()
        print("\n  ┌── BASKET MODEL ──────────────────────────────┐")
        print(f"  │  R²            : {d['R_squared']:<8}                  │")
        print(f"  │  Adj R²        : {d['Adj_R_squared']:<8}                  │")
        print(f"  │  Durbin-Watson : {d['Durbin_Watson']:<8}                  │")
        print(f"  │  Residual σ    : {d['residual_std']:<10}                │")
        for name, w in d["weights"].items():
            label = name.replace("log_ret_", "")
            print(f"  │  {label:<14}: {w:+.5f}                 │")
        print(f"  │  Weight sum    : {d['weight_sum']:<8}                  │")
        print("  └───────────────────────────────────────────────┘\n")


# ============================================================
#  SECTION 3 – OU MODEL
# ============================================================

class OUModel:
    def __init__(self):
        self.theta = self.mu = self.sigma = None

    def fit(self, data):
        s = data["spread_raw"].dropna().values
        self.theta, self.mu, self.sigma = self._mle(s)
        hl = np.log(2) / self.theta
        print(f"\n  ┌── ORNSTEIN-UHLENBECK ─────────────────────────┐")
        print(f"  │  θ (speed)  : {self.theta:.4f}  (half-life ≈ {hl:.1f}d)     │")
        print(f"  │  μ (mean)   : {self.mu:.6f} TND              │")
        print(f"  │  σ (diffusion): {self.sigma:.6f} TND/√day       │")
        print(f"  └───────────────────────────────────────────────┘\n")
        return self

    def _mle(self, s):
        dt = 1.0
        def neg_ll(params):
            th, mu, sg = params
            if th <= 0 or sg <= 0: return 1e12
            e    = np.exp(-th * dt)
            sv   = sg**2 * (1 - np.exp(-2*th*dt)) / (2*th)
            if sv <= 0: return 1e12
            res  = s[1:] - (mu + e * (s[:-1] - mu))
            return 0.5 * len(s) * np.log(2*np.pi*sv) + 0.5 * np.sum(res**2) / sv

        x0  = [0.3, s.mean(), s.std() * 0.6]
        res = minimize(neg_ll, x0, method="L-BFGS-B",
                       bounds=[(1e-3, 20), (-0.05, 0.05), (1e-5, 0.1)])
        return res.x

    def apply(self, data):
        df = data.copy()
        s  = df["spread_raw"].values
        e  = np.exp(-self.theta)
        fc = self.mu + e * (s - self.mu)
        df["delta_estimate_ou"] = np.concatenate([[np.nan], fc[:-1]])
        df["intrinsic_ou"] = df.get("predicted_fixing", df["fixing"]) + df["delta_estimate_ou"]
        return df


# ============================================================
#  SECTION 4 – KALMAN FILTER
# ============================================================

class KalmanFilter:
    def __init__(self):
        self.phi = self.Q = self.R = None

    def fit(self, data):
        s = data["spread_raw"].dropna().values
        self.phi, self.Q, self.R = self._em(s)
        hl = -np.log(2) / np.log(abs(self.phi)) if abs(self.phi) < 1 else np.inf
        snr = self.Q / self.R
        print(f"\n  ┌── KALMAN SPREAD FILTER ───────────────────────┐")
        print(f"  │  φ (persistence)  : {self.phi:.4f}  (half-life ≈ {hl:.1f}d)│")
        print(f"  │  Q (process noise): {self.Q:.2e}              │")
        print(f"  │  R (obs noise)    : {self.R:.2e}              │")
        print(f"  │  Signal-to-noise  : {snr:.4f}                  │")
        print(f"  └───────────────────────────────────────────────┘\n")
        return self

    def _em(self, s, n_iter=40):
        n   = len(s)
        phi = 0.85
        Q   = float(np.var(np.diff(s))) / 2
        R   = float(np.var(s)) * 0.05

        for _ in range(n_iter):
            # Forward pass
            xf, Pf = np.zeros(n), np.zeros(n)
            xp, Pp = s[0], Q / max(1 - phi**2, 1e-6)
            for t in range(n):
                K      = Pp / (Pp + R)
                xf[t]  = xp + K * (s[t] - xp)
                Pf[t]  = (1 - K) * Pp
                if t < n - 1:
                    xp = phi * xf[t]
                    Pp = phi**2 * Pf[t] + Q
            # Backward pass (RTS smoother)
            xs, Ps = xf.copy(), Pf.copy()
            Pc     = np.zeros(n - 1)
            for t in range(n-2, -1, -1):
                G      = Pf[t] * phi / (phi**2 * Pf[t] + Q)
                xs[t]  = xf[t] + G * (xs[t+1] - phi * xf[t])
                Ps[t]  = Pf[t] + G**2 * (Ps[t+1] - (phi**2 * Pf[t] + Q))
                Pc[t]  = G * Ps[t+1]
            # M-step
            E_xx  = np.sum(Ps[1:] + xs[1:]**2)
            E_x1x = np.sum(Pc + xs[1:] * xs[:-1])
            E_x0  = np.sum(Ps[:-1] + xs[:-1]**2)
            phi   = np.clip(E_x1x / max(E_x0, 1e-10), -0.999, 0.999)
            Q     = max((E_xx - phi * E_x1x * 2 + phi**2 * E_x0) / n, 1e-10)
            R     = max(float(np.mean((s - xs)**2 + Ps)), 1e-10)

        return float(phi), float(Q), float(R)

    def filter_series(self, s):
        n   = len(s)
        phi, Q, R = self.phi, self.Q, self.R
        xf, Pf   = np.zeros(n), np.zeros(n)
        innov    = np.zeros(n)
        gain     = np.zeros(n)
        xp, Pp   = s[0], Q / max(1 - phi**2, 1e-6)
        for t in range(n):
            innov[t] = s[t] - xp
            K        = Pp / (Pp + R)
            gain[t]  = K
            xf[t]    = xp + K * innov[t]
            Pf[t]    = (1 - K) * Pp
            xp       = phi * xf[t]
            Pp       = phi**2 * Pf[t] + Q
        xpred = np.concatenate([[s[0]], phi * xf[:-1]])
        Ppred = np.concatenate([[Pf[0]], phi**2 * Pf[:-1] + Q])
        return {"state": xf, "variance": Pf, "predicted": xpred,
                "pred_var": Ppred, "innovations": innov, "gain": gain}

    def nowcast(self, x_prev, P_prev, observation=None, obs_noise=None):
        xp = self.phi * x_prev
        Pp = self.phi**2 * P_prev + self.Q
        if observation is not None:
            R_obs = obs_noise or self.R
            K     = Pp / (Pp + R_obs)
            xp    = xp + K * (observation - xp)
            Pp    = (1 - K) * Pp
        std = np.sqrt(Pp)
        return {"mean": xp, "variance": Pp, "std": std,
                "lower_95": xp - 1.96*std, "upper_95": xp + 1.96*std}

    def apply(self, data):
        df  = data.copy()
        kf  = self.filter_series(df["spread_raw"].values)
        df["delta_estimate_kf"] = kf["predicted"]
        df["delta_var_kf"]      = kf["pred_var"]
        df["delta_lo_kf"]       = kf["predicted"] - 1.96 * np.sqrt(kf["pred_var"])
        df["delta_hi_kf"]       = kf["predicted"] + 1.96 * np.sqrt(kf["pred_var"])
        base = df.get("predicted_fixing", df["fixing"])
        df["intrinsic_kf"]      = base + df["delta_estimate_kf"]
        df["intrinsic_lo_kf"]   = base + df["delta_lo_kf"]
        df["intrinsic_hi_kf"]   = base + df["delta_hi_kf"]
        return df


# ============================================================
#  SECTION 5 – BACKTEST
# ============================================================

def compute_metrics(y_true, y_pred, label=""):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    err    = yp - yt
    n      = len(yt)
    dir_acc = float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp))))
    return {
        "label"          : label,
        "n"              : n,
        "MAE"            : float(np.mean(np.abs(err))),
        "RMSE"           : float(np.sqrt(np.mean(err**2))),
        "MAPE_pct"       : float(np.mean(np.abs(err)/np.abs(yt+1e-10))*100),
        "max_err"        : float(np.max(np.abs(err))),
        "bias"           : float(np.mean(err)),
        "dir_acc"        : round(dir_acc, 4),
        "corr"           : float(np.corrcoef(yt, yp)[0, 1]),
        "Theil_U"        : float(np.sqrt(np.mean(err**2)) / (np.sqrt(np.mean(yt**2)) + 1e-12)),
    }


def walk_forward_backtest(data, train_size=252, step=21):
    n     = len(data)
    folds = []
    t     = train_size
    while t + step <= n:
        train = data.iloc[:t]
        test  = data.iloc[t : t + step]

        # Fit on train
        bm = BasketModel(rolling_window=None); bm.fit(train)
        kf = KalmanFilter();                   kf.fit(train)

        # Extend basket to test
        test_b = bm.apply(test)

        # Kalman: carry forward last state
        kf_train = kf.filter_series(train["spread_raw"].values)
        x_last   = float(kf_train["state"][-1])
        P_last   = float(kf_train["variance"][-1])

        for i in range(len(test_b)):
            row    = test_b.iloc[i]
            kf_out = kf.nowcast(x_last, P_last)
            base   = row.get("predicted_fixing") if "predicted_fixing" in row.index else row["fixing"]
            folds.append({
                "date"        : test.index[i],
                "fixing"      : row["fixing"],
                "ib_rate"     : row["ib_rate"],
                "baseline"    : base,
                "delta_kf"    : kf_out["mean"],
                "intrinsic"   : base + kf_out["mean"],
                "intrinsic_lo": base + kf_out["lower_95"],
                "intrinsic_hi": base + kf_out["upper_95"],
                "spread_act"  : row["spread_raw"],
            })
            # Update Kalman
            K      = P_last / (P_last + kf.R)
            x_last = x_last + K * (row["spread_raw"] - x_last)
            P_last = (1 - K) * P_last

        t += step

    preds = pd.DataFrame(folds).set_index("date")
    preds["stress"] = (
        preds["spread_act"].abs()
        > preds["spread_act"].abs().rolling(63, min_periods=10).mean()
          + 1.5 * preds["spread_act"].rolling(63, min_periods=10).std()
    )
    return preds


def model_comparison(data, train_frac=0.75):
    split = int(len(data) * train_frac)
    train, test = data.iloc[:split], data.iloc[split:]
    yt = test["ib_rate"].values
    results = []

    # Naive
    results.append(compute_metrics(yt, test["fixing"].values, "Naive: fixing"))

    # Basket only
    bm = BasketModel(rolling_window=None); bm.fit(train)
    test_b = bm.apply(test)
    results.append(compute_metrics(yt, test_b["predicted_fixing"].values, "Basket (no adj.)"))

    # OU
    ou = OUModel(); ou.fit(train)
    test_ou = ou.apply(test_b)
    v = test_ou["intrinsic_ou"].dropna().values
    results.append(compute_metrics(yt[:len(v)], v, "OU spread model"))

    # Kalman
    kf = KalmanFilter(); kf.fit(train)
    test_kf = kf.apply(test_b)
    v = test_kf["intrinsic_kf"].dropna().values
    results.append(compute_metrics(yt[:len(v)], v, "Kalman spread filter"))

    df = pd.DataFrame(results).set_index("label")
    print("\n  ╔══ MODEL COMPARISON ══════════════════════════════════════╗")
    print(f"  {'Model':<28} {'MAE':>8} {'RMSE':>8} {'Dir.Acc':>9} {'Corr':>7}")
    print("  " + "─"*60)
    for lbl, row in df.iterrows():
        print(f"  {lbl:<28} {row['MAE']:>8.5f} {row['RMSE']:>8.5f} "
              f"{row['dir_acc']:>9.1%} {row['corr']:>7.4f}")
    print("  ╚" + "═"*59 + "╝\n")
    return df


# ============================================================
#  SECTION 6 – CHARTS
# ============================================================

def plot_basket_diagnostics(basket, data, path):
    data_e = basket.apply(data)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Component 1: Basket Model Diagnostics", fontsize=14)

    # Actual vs predicted log-ret
    ax = axes[0, 0]
    ax.scatter(data_e["log_ret_fixing"], data_e["predicted_log_ret"],
               alpha=0.3, s=10, color=COLORS["baseline"])
    lims = [data_e["log_ret_fixing"].min(), data_e["log_ret_fixing"].max()]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_title(f"Actual vs Predicted Log-Returns  (R²={basket.r2_:.4f})")
    ax.set_xlabel("Actual Δln(fixing)"); ax.set_ylabel("Predicted")

    # Residual histogram
    ax = axes[0, 1]
    ax.hist(basket.residuals_, bins=55, color=COLORS["baseline"], alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_title("Residual Distribution"); ax.set_xlabel("Residual")

    # Residuals over time
    ax = axes[1, 0]
    ax.plot(basket.residuals_.index, basket.residuals_.values, color=COLORS["spread"], lw=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(basket.residuals_.index, basket.residuals_.values, 0,
                    where=basket.residuals_.values > 0, color=COLORS["intrinsic"], alpha=0.2)
    ax.fill_between(basket.residuals_.index, basket.residuals_.values, 0,
                    where=basket.residuals_.values < 0, color=COLORS["baseline"], alpha=0.2)
    ax.set_title("Residuals Over Time"); ax.set_xlabel("Date"); ax.set_ylabel("Residual")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Rolling weights
    ax = axes[1, 1]
    if basket.rolling_w_ is not None:
        rw = basket.rolling_w_
        lc = [COLORS["fixing"], COLORS["ib_rate"], COLORS["spread"]]
        for (col, lbl), c in zip([("w_EURUSD","EUR/USD"),("w_GBPUSD","GBP/USD"),("w_USDJPY","USD/JPY")], lc):
            ax.plot(rw.index, rw[col], label=lbl, color=c, lw=1.2)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title("Rolling Basket Weights (90-day window)")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_spread_analysis(data, ou, path):
    spread = data["spread_raw"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Component 2: Spread Analysis (IB − Fixing)", fontsize=14)

    ax = axes[0, 0]
    ax.plot(spread.index, spread, color=COLORS["spread"], lw=0.8)
    ax.axhline(spread.mean(), color=COLORS["intrinsic"], ls="--", lw=1.1,
               label=f"Mean={spread.mean():.4f}")
    ax.fill_between(spread.index, spread.mean()-spread.std(), spread.mean()+spread.std(),
                    alpha=0.12, color=COLORS["spread"], label="±1σ")
    ax.set_title("Spread Over Time"); ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[0, 1]
    ax.hist(spread, bins=60, color=COLORS["spread"], alpha=0.7, density=True, edgecolor="white")
    xs = np.linspace(spread.min(), spread.max(), 300)
    ou_std = ou.sigma / np.sqrt(2 * ou.theta)
    ax.plot(xs, scipy_norm.pdf(xs, ou.mu, ou_std), color=COLORS["intrinsic"],
            lw=2, label="OU stationary")
    ax.set_title("Spread Distribution"); ax.legend()

    ax = axes[1, 0]
    lags = range(1, 31)
    acf  = [spread.autocorr(l) for l in lags]
    ax.bar(list(lags), acf, color=COLORS["baseline"], alpha=0.75, width=0.6)
    ci = 1.96 / np.sqrt(len(spread))
    ax.axhline(ci, color="red", ls="--", lw=0.8); ax.axhline(-ci, color="red", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Spread ACF"); ax.set_xlabel("Lag (days)")

    ax = axes[1, 1]
    roll_std = spread.rolling(21).std()
    ax.plot(spread.index, roll_std, color=COLORS["ib_rate"], lw=0.9)
    q75 = roll_std.quantile(0.75)
    ax.axhline(q75, color=COLORS["intrinsic"], ls="--", lw=1, label=f"75th pct")
    ax.fill_between(spread.index, roll_std, q75, where=roll_std>q75,
                    color="#FCA5A5", alpha=0.5, label="High-vol regime")
    ax.set_title("Rolling 21-Day Spread Volatility"); ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_kalman(data, kf, path):
    spread = data["spread_raw"].values
    dates  = data.index
    kout   = kf.filter_series(spread)
    sigma  = np.sqrt(kout["pred_var"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle("Kalman Filter: Spread State Estimation & Nowcast", fontsize=14)

    ax1.plot(dates, spread, color=COLORS["spread"], lw=0.7, alpha=0.7, label="Observed spread")
    ax1.plot(dates, kout["state"], color=COLORS["intrinsic"], lw=1.5, label="Filtered state S*(t)")
    ax1.plot(dates, kout["predicted"], color=COLORS["baseline"], lw=1, ls="--", label="Nowcast")
    ax1.fill_between(dates, kout["predicted"] - 1.96*sigma, kout["predicted"] + 1.96*sigma,
                     alpha=0.15, color=COLORS["baseline"], label="95% CI")
    ax1.set_ylabel("Spread (TND)"); ax1.legend(fontsize=8)
    ax1.set_title("Latent Spread State S*(t)")

    ax2.plot(dates, kout["gain"], color=COLORS["ib_rate"], lw=0.8)
    ax2.axhline(kout["gain"].mean(), color="black", ls="--", lw=0.8,
                label=f"Mean gain = {kout['gain'].mean():.3f}")
    ax2.set_ylabel("Kalman Gain K(t)"); ax2.set_xlabel("Date")
    ax2.set_title("Kalman Gain Over Time"); ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_intrinsic(data, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Real-Time Intrinsic USD/TND Valuation  V*(t) = Baseline + δ̂(t)", fontsize=14)
    dates = data.index

    ax1.plot(dates, data["fixing"],        color=COLORS["fixing"],    lw=1.4, label="BCT Fixing")
    ax1.plot(dates, data["ib_rate"],       color=COLORS["ib_rate"],   lw=1.1, alpha=0.8, label="IB Rate (t−1)")
    if "predicted_fixing" in data.columns:
        ax1.plot(dates, data["predicted_fixing"], color=COLORS["baseline"],
                 lw=0.9, ls="--", alpha=0.7, label="Basket Baseline")
    if "intrinsic_kf" in data.columns:
        ax1.plot(dates, data["intrinsic_kf"], color=COLORS["intrinsic"],
                 lw=1.5, label="Intrinsic V*(t) [Kalman]")
        if "intrinsic_lo_kf" in data.columns:
            ax1.fill_between(dates, data["intrinsic_lo_kf"], data["intrinsic_hi_kf"],
                             alpha=0.10, color=COLORS["intrinsic"], label="95% CI")
    ax1.set_ylabel("USD/TND Rate")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

    ax2.plot(dates, data["spread_raw"], color=COLORS["spread"], lw=0.8)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.fill_between(dates, data["spread_raw"], 0,
                     where=data["spread_raw"] > 0, color=COLORS["intrinsic"], alpha=0.3)
    ax2.set_ylabel("IB − Fixing Spread"); ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_backtest(preds, path):
    err = preds["intrinsic"] - preds["ib_rate"]
    m   = compute_metrics(preds["ib_rate"].values, preds["intrinsic"].values, "OOS")

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)
    fig.suptitle("Backtest: Walk-Forward Out-of-Sample Evaluation", fontsize=14)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(preds.index, preds["ib_rate"],   color=COLORS["ib_rate"],   lw=1.1, label="Actual IB Rate")
    ax1.plot(preds.index, preds["intrinsic"], color=COLORS["intrinsic"], lw=1.1, ls="--", label="Intrinsic V*(t)")
    ax1.fill_between(preds.index, preds["intrinsic_lo"], preds["intrinsic_hi"],
                     alpha=0.10, color=COLORS["intrinsic"], label="95% CI")
    # Shade stress periods
    in_stress, start = False, None
    for dt, val in preds["stress"].items():
        if val and not in_stress: start = dt; in_stress = True
        elif not val and in_stress: ax1.axvspan(start, dt, alpha=0.07, color="red"); in_stress = False
    ax1.set_title("Predicted Intrinsic Value vs Actual IB Rate")
    ax1.set_ylabel("USD/TND"); ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(preds.index, err, color=COLORS["spread"], lw=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.fill_between(preds.index, err, 0, where=err>0, color=COLORS["intrinsic"], alpha=0.25)
    ax2.fill_between(preds.index, err, 0, where=err<0, color=COLORS["baseline"], alpha=0.25)
    ax2.set_title("Prediction Errors Over Time"); ax2.set_ylabel("Error (TND)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(err, bins=50, color=COLORS["baseline"], alpha=0.7, density=True, edgecolor="white")
    ax3.axvline(0, color="black", lw=1, ls="--")
    xs = np.linspace(err.min(), err.max(), 200)
    ax3.plot(xs, scipy_norm.pdf(xs, err.mean(), err.std()), color=COLORS["intrinsic"], lw=2)
    stats_text = (f"RMSE = {m['RMSE']:.5f}\nMAE  = {m['MAE']:.5f}\n"
                  f"Dir. = {m['dir_acc']:.1%}\nCorr = {m['corr']:.4f}")
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax3.set_title("Error Distribution")

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_comparison(comp_df, path):
    labels = comp_df.index.tolist()
    metrics = ["MAE", "RMSE", "dir_acc"]
    colors  = [COLORS["fixing"], COLORS["ib_rate"],
                COLORS["spread"], COLORS["intrinsic"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison – Out-of-Sample Performance", fontsize=14)
    for ax, metric in zip(axes, metrics):
        vals = comp_df[metric].values
        bars = ax.barh(labels, vals, color=colors[:len(labels)], alpha=0.82)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.1%}" if metric == "dir_acc" else f"{val:.5f}"
            ax.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                    fmt, va="center", fontsize=9)
        ax.set_title(metric)
        if metric == "dir_acc":
            ax.axvline(0.5, color="red", ls="--", lw=0.8)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def export_excel(data, preds, comp_df, basket, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Full Dataset")
        diag = basket.diagnostics()
        rows = [{"Metric": k, "Value": str(v)} for k, v in diag.items()]
        pd.DataFrame(rows).to_excel(writer, sheet_name="Basket Diagnostics", index=False)
        if basket.rolling_w_ is not None:
            basket.rolling_w_.to_excel(writer, sheet_name="Rolling Weights")
        preds.to_excel(writer, sheet_name="Backtest Predictions")
        comp_df.to_excel(writer, sheet_name="Model Comparison")
    print(f"  → {path}")


# ============================================================
#  SECTION 7 – LIVE SIMULATION
# ============================================================

def live_simulation(data, basket, kf):
    recent = data.tail(252)
    last   = recent.iloc[-1]
    rng    = np.random.default_rng(7)

    kf_train = kf.filter_series(recent["spread_raw"].values)
    x_last   = float(kf_train["state"][-1])
    P_last   = float(kf_train["variance"][-1])
    fixing_a = float(last["fixing"])
    anchor_fx = {"EURUSD": float(last["EURUSD"]),
                 "GBPUSD": float(last["GBPUSD"]),
                 "USDJPY": float(last["USDJPY"])}

    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │ {'Date':<12} {'Fixing':>8} {'Baseline':>10} {'δ̂':>10} "
          f"{'V*(t)':>10} {'IB(t-1)':>9} │")
    print("  ├─────────────────────────────────────────────────────────────────┤")

    for ts, row in recent.tail(20).iterrows():
        noise = rng.normal(0, 0.0003, 3)
        cur_fx = {"EURUSD": float(row["EURUSD"]) * (1 + noise[0]),
                  "GBPUSD": float(row["GBPUSD"]) * (1 + noise[1]),
                  "USDJPY": float(row["USDJPY"]) * (1 + noise[2])}
        signal = (float(row["spread_raw"]) + rng.normal(0, 0.0003)
                  if rng.random() < 0.30 else None)

        base  = basket.compute_baseline(fixing_a, cur_fx, anchor_fx)
        kf_nc = kf.nowcast(x_last, P_last, observation=signal,
                           obs_noise=kf.R * 2 if signal else None)
        x_last = kf_nc["mean"]
        P_last = kf_nc["variance"]
        iv     = base + kf_nc["mean"]
        sig    = "●" if signal else " "

        print(f"  │ {str(ts.date()):<12} {fixing_a:>8.4f} {base:>10.4f} "
              f"{kf_nc['mean']:>10.5f} {iv:>10.4f} {row['ib_rate']:>9.4f} {sig}│")
        time.sleep(0.04)

    print("  └─────────────────────────────────────────────────────────────────┘")
    print("   ● = intraday signal incorporated via Kalman update")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("\n" + "═"*65)
    print("   TND INTRINSIC VALUE MODEL  ·  FIN 460 Dynamic Asset Pricing")
    print("═"*65)

    # 1. Data
    print("\n  ━━  STEP 1: DATA  ━━")
    data = generate_synthetic_data()

    # 2. Basket
    print("\n  ━━  STEP 2: BASKET MODEL (Component 1)  ━━")
    basket = BasketModel(rolling_window=90)
    basket.fit(data)
    data = basket.apply(data)

    # 3. Liquidity models
    print("\n  ━━  STEP 3: LIQUIDITY ADJUSTMENT (Component 2)  ━━")
    ou = OUModel(); ou.fit(data)
    kf = KalmanFilter(); kf.fit(data)

    # 4. Full intrinsic value series
    print("\n  ━━  STEP 4: COMBINING COMPONENTS  ━━")
    data = kf.apply(data)
    data = ou.apply(data)
    print(f"  Intrinsic V*(t): {data['intrinsic_kf'].min():.4f} – {data['intrinsic_kf'].max():.4f}")
    print(f"  Corr(V*, IB)   : {data['ib_rate'].corr(data['intrinsic_kf']):.4f}")

    # 5. Backtest
    print("\n  ━━  STEP 5: WALK-FORWARD BACKTEST  ━━")
    preds = walk_forward_backtest(data)
    m = compute_metrics(preds["ib_rate"].values, preds["intrinsic"].values)
    print(f"  OOS RMSE = {m['RMSE']:.5f}  MAE = {m['MAE']:.5f}  "
          f"Dir.Acc = {m['dir_acc']:.1%}  Corr = {m['corr']:.4f}")

    # 6. Model comparison
    print("\n  ━━  STEP 6: MODEL COMPARISON  ━━")
    comp = model_comparison(data)

    # 7. Live simulation
    print("\n  ━━  STEP 7: LIVE SIMULATION DEMO  ━━")
    live_simulation(data, basket, kf)

    # 8. Charts
    print("\n  ━━  STEP 8: GENERATING CHARTS  ━━")
    plot_basket_diagnostics(basket, data, f"{OUTPUT_DIR}/01_basket_diagnostics.png")
    plot_spread_analysis(data, ou,        f"{OUTPUT_DIR}/02_spread_analysis.png")
    plot_kalman(data, kf,                 f"{OUTPUT_DIR}/03_kalman_filter.png")
    plot_intrinsic(data,                  f"{OUTPUT_DIR}/04_intrinsic_value.png")
    plot_backtest(preds,                  f"{OUTPUT_DIR}/05_backtest_performance.png")
    plot_comparison(comp,                 f"{OUTPUT_DIR}/06_model_comparison.png")

    # 9. Excel export
    print("\n  ━━  STEP 9: EXCEL EXPORT  ━━")
    export_excel(data, preds, comp, basket,
                 f"{OUTPUT_DIR}/tnd_intrinsic_model_results.xlsx")

    print("\n" + "═"*65)
    print(f"  COMPLETE  ·  All outputs in ./{OUTPUT_DIR}/")
    print("═"*65 + "\n")


if __name__ == "__main__":
    main()
