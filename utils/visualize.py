"""
utils/visualize.py
------------------
All charts for the TND Intrinsic Value Model.

Functions
---------
plot_basket_diagnostics      – weight stability, residuals, R² over time
plot_spread_analysis         – historical spread distribution and OU fit
plot_kalman_filter           – filtered state + 95% CI vs actual spread
plot_intrinsic_value         – main output chart: fixing, IB, intrinsic
plot_backtest_performance    – OOS predictions vs realised IB rate
plot_model_comparison        – metrics bar chart for all models
plot_regime_analysis         – stress vs normal period performance
save_all_charts              – convenience: render & save everything
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

COLORS = {
    "fixing"    : "#2563EB",   # blue
    "ib_rate"   : "#059669",   # green
    "intrinsic" : "#DC2626",   # red
    "baseline"  : "#7C3AED",   # purple
    "spread"    : "#D97706",   # amber
    "band"      : "#FEE2E2",   # light red
    "stress"    : "#FCA5A5",
    "grid"      : "#E5E7EB",
}

def _style():
    plt.rcParams.update({
        "figure.facecolor" : "white",
        "axes.facecolor"   : "white",
        "axes.grid"        : True,
        "grid.color"       : COLORS["grid"],
        "grid.linewidth"   : 0.5,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "font.size"        : 11,
        "axes.titlesize"   : 13,
        "axes.titleweight" : "bold",
        "axes.labelsize"   : 10,
        "legend.fontsize"  : 9,
        "figure.dpi"       : 130,
    })

_style()


# ---------------------------------------------------------------------------
# 1. Basket model diagnostics
# ---------------------------------------------------------------------------

def plot_basket_diagnostics(basket_model, data: pd.DataFrame,
                             save_path: str = None):
    result = basket_model.result_
    fig    = plt.figure(figsize=(16, 10))
    gs     = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # (a) Actual vs predicted log-returns
    ax1 = fig.add_subplot(gs[0, :2])
    data_ext = basket_model.apply_to_dataset(data)
    ax1.scatter(data_ext["log_ret_fixing"],
                data_ext["predicted_log_ret"],
                alpha=0.35, s=12, color=COLORS["baseline"])
    lims = [data_ext["log_ret_fixing"].min(), data_ext["log_ret_fixing"].max()]
    ax1.plot(lims, lims, "k--", lw=1, label="Perfect fit")
    ax1.set_xlabel("Actual Δln(fixing)")
    ax1.set_ylabel("Predicted Δln(fixing)")
    ax1.set_title("Actual vs Predicted Log-Returns (Basket)")
    ax1.legend()
    r2_text = f"R² = {result.r_squared:.4f}"
    ax1.text(0.05, 0.92, r2_text, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # (b) Residuals over time
    ax2 = fig.add_subplot(gs[0, 2])
    resid = result.residuals
    ax2.hist(resid, bins=50, color=COLORS["baseline"], alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="black", lw=1, ls="--")
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    std_text = f"σ = {resid.std():.5f}"
    ax2.text(0.65, 0.92, std_text, transform=ax2.transAxes, fontsize=9)

    # (c) Residuals in time
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(resid.index, resid.values, color=COLORS["spread"], lw=0.7, alpha=0.8)
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.fill_between(resid.index, resid.values, 0,
                     where=resid.values > 0, color=COLORS["intrinsic"], alpha=0.2)
    ax3.fill_between(resid.index, resid.values, 0,
                     where=resid.values < 0, color=COLORS["baseline"], alpha=0.2)
    ax3.set_title("Basket Model Residuals Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Residual (log-return)")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # (d) Rolling weights
    ax4 = fig.add_subplot(gs[1, 2])
    if result.rolling_weights is not None and not result.rolling_weights.empty:
        rw = result.rolling_weights
        w_labels = {"w_EURUSD": "EUR/USD", "w_GBPUSD": "GBP/USD", "w_USDJPY": "USD/JPY"}
        colors_rw = [COLORS["fixing"], COLORS["ib_rate"], COLORS["spread"]]
        for (col, label), color in zip(w_labels.items(), colors_rw):
            if col in rw.columns:
                ax4.plot(rw.index, rw[col], label=label, color=color, lw=1.2)
        ax4.axhline(0, color="black", lw=0.6, ls="--")
        ax4.set_title("Rolling Basket Weights (90-day)")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Weight")
        ax4.legend()
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax4.text(0.5, 0.5, "Rolling weights\nnot computed",
                 ha="center", va="center", transform=ax4.transAxes)

    fig.suptitle("Basket Model Diagnostics – Component 1", fontsize=14, y=1.01)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Spread analysis
# ---------------------------------------------------------------------------

def plot_spread_analysis(data: pd.DataFrame, ou_model=None,
                          save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Spread Analysis: IB Rate − BCT Fixing  (Component 2)",
                 fontsize=14, y=1.01)

    spread = data["spread_raw"]

    # (a) Spread over time
    ax = axes[0, 0]
    ax.plot(spread.index, spread.values, color=COLORS["spread"], lw=0.8)
    ax.axhline(spread.mean(), color=COLORS["intrinsic"], lw=1.2, ls="--",
               label=f"Mean = {spread.mean():.4f}")
    ax.fill_between(spread.index,
                    spread.mean() - spread.std(),
                    spread.mean() + spread.std(),
                    alpha=0.15, color=COLORS["spread"], label="±1σ")
    ax.set_title("Historical Spread: IB − Fixing")
    ax.set_ylabel("Spread (TND)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # (b) Spread distribution with OU fit
    ax = axes[0, 1]
    ax.hist(spread.values, bins=60, color=COLORS["spread"], alpha=0.65,
            density=True, edgecolor="white", label="Observed")
    if ou_model and ou_model.params_:
        p      = ou_model.params_
        xs     = np.linspace(spread.min(), spread.max(), 300)
        # Stationary distribution of OU is Normal(μ, σ²/(2θ))
        ou_std = p.sigma / np.sqrt(2 * p.theta)
        from scipy.stats import norm
        ax.plot(xs, norm.pdf(xs, p.mu, ou_std),
                color=COLORS["intrinsic"], lw=2, label="OU stationary dist.")
    ax.set_title("Spread Distribution")
    ax.set_xlabel("Spread (TND)")
    ax.set_ylabel("Density")
    ax.legend()

    # (c) Spread autocorrelation
    ax = axes[1, 0]
    lags   = np.arange(1, 31)
    acf_vals = [spread.autocorr(lag=int(l)) for l in lags]
    ax.bar(lags, acf_vals, color=COLORS["baseline"], alpha=0.7, width=0.6)
    conf = 1.96 / np.sqrt(len(spread))
    ax.axhline(conf,  color="red", ls="--", lw=0.8, label="95% CI")
    ax.axhline(-conf, color="red", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Spread Autocorrelation")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.legend()

    # (d) Spread by rolling volatility (regime detection proxy)
    ax = axes[1, 1]
    roll_std = spread.rolling(21).std()
    ax.plot(spread.index, roll_std.values, color=COLORS["ib_rate"], lw=0.9)
    q75 = roll_std.quantile(0.75)
    ax.axhline(q75, color=COLORS["intrinsic"], ls="--", lw=1,
               label=f"75th pctile ({q75:.4f})")
    ax.fill_between(spread.index, roll_std.values, q75,
                    where=roll_std.values > q75,
                    color=COLORS["stress"], alpha=0.5, label="High-vol regime")
    ax.set_title("Rolling 21-Day Spread Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread σ")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Kalman filter chart
# ---------------------------------------------------------------------------

def plot_kalman_filter(data: pd.DataFrame, kalman_model,
                        save_path: str = None):
    kf_out = kalman_model.filter_series(data["spread_raw"].values)
    dates  = data.index
    sigma  = np.sqrt(kf_out["pred_var"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle("Kalman Filter – Spread State Estimation & Nowcast", fontsize=14)

    # Upper: filtered state vs actual spread
    ax1.plot(dates, data["spread_raw"].values, color=COLORS["spread"],
             lw=0.8, alpha=0.7, label="Observed spread (t−1 lag)")
    ax1.plot(dates, kf_out["state"], color=COLORS["intrinsic"],
             lw=1.5, label="Kalman filtered state S*(t)")
    ax1.plot(dates, kf_out["predicted"], color=COLORS["baseline"],
             lw=1, ls="--", label="One-step-ahead nowcast")
    ax1.fill_between(dates,
                     kf_out["predicted"] - 1.96 * sigma,
                     kf_out["predicted"] + 1.96 * sigma,
                     alpha=0.18, color=COLORS["baseline"], label="95% nowcast CI")
    ax1.set_ylabel("Spread (TND)")
    ax1.set_title("State Estimation: Latent Spread S*(t)")
    ax1.legend(fontsize=8)

    # Lower: Kalman gain over time
    ax2.plot(dates, kf_out["kalman_gain"], color=COLORS["ib_rate"], lw=0.8)
    ax2.axhline(kf_out["kalman_gain"].mean(), color="black", ls="--", lw=0.8,
                label=f"Mean gain = {kf_out['kalman_gain'].mean():.3f}")
    ax2.set_ylabel("Kalman Gain K(t)")
    ax2.set_xlabel("Date")
    ax2.set_title("Kalman Gain (signal weight on new observations)")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Main intrinsic value chart
# ---------------------------------------------------------------------------

def plot_intrinsic_value(data: pd.DataFrame, save_path: str = None):
    """
    Primary output chart: BCT fixing, IB rate, basket baseline, and
    intrinsic value (Kalman-adjusted) on the same panel.
    """
    required = ["fixing", "ib_rate"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' missing from data.")

    has_intrinsic = "intrinsic_kf" in data.columns
    has_baseline  = "predicted_fixing" in data.columns

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Real-Time Intrinsic USD/TND Valuation", fontsize=15)

    dates = data.index

    # --- Main panel ---
    ax1.plot(dates, data["fixing"],  color=COLORS["fixing"],
             lw=1.4, label="BCT Official Fixing", zorder=4)
    ax1.plot(dates, data["ib_rate"], color=COLORS["ib_rate"],
             lw=1.2, alpha=0.85, label="Interbank Rate (t−1)", zorder=3)

    if has_baseline:
        ax1.plot(dates, data["predicted_fixing"], color=COLORS["baseline"],
                 lw=1.0, ls="--", alpha=0.8, label="Basket Baseline", zorder=2)

    if has_intrinsic:
        ax1.plot(dates, data["intrinsic_kf"], color=COLORS["intrinsic"],
                 lw=1.5, label="Intrinsic Value V*(t) [Kalman]", zorder=5)
        if "delta_upper_95_kf" in data.columns:
            ax1.fill_between(dates,
                             data["delta_lower_95_kf"] + data.get("predicted_fixing",
                                                                    data["fixing"]),
                             data["delta_upper_95_kf"] + data.get("predicted_fixing",
                                                                    data["fixing"]),
                             alpha=0.1, color=COLORS["intrinsic"], label="95% CI")

    ax1.set_ylabel("USD/TND Rate")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

    # --- Lower panel: spread ---
    ax2.plot(dates, data["spread_raw"], color=COLORS["spread"], lw=0.8)
    ax2.axhline(0, color="black", lw=0.7, ls="-")
    ax2.axhline(data["spread_raw"].mean(), color=COLORS["spread"],
                lw=0.8, ls="--", alpha=0.6)
    ax2.fill_between(dates, data["spread_raw"], 0,
                     where=data["spread_raw"] > 0,
                     color=COLORS["intrinsic"], alpha=0.3, label="IB premium")
    ax2.fill_between(dates, data["spread_raw"], 0,
                     where=data["spread_raw"] <= 0,
                     color=COLORS["baseline"], alpha=0.3, label="IB discount")
    ax2.set_ylabel("IB − Fixing Spread")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Backtest performance
# ---------------------------------------------------------------------------

def plot_backtest_performance(backtest_result, save_path: str = None):
    preds = backtest_result.predictions
    m     = backtest_result.metrics_oos

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    fig.suptitle("Backtest Performance – Out-of-Sample Evaluation", fontsize=14)

    # (a) Predicted vs actual IB rate
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(preds.index, preds["ib_rate"],   color=COLORS["ib_rate"],
             lw=1.1, label="Actual IB Rate")
    ax1.plot(preds.index, preds["intrinsic"], color=COLORS["intrinsic"],
             lw=1.1, ls="--", label="Intrinsic Value Prediction")
    ax1.fill_between(preds.index, preds["intrinsic_lo"], preds["intrinsic_hi"],
                     alpha=0.12, color=COLORS["intrinsic"], label="95% CI")
    # Shade stress periods
    if "stress_regime" in preds.columns:
        for start, end in _get_stress_periods(preds):
            ax1.axvspan(start, end, alpha=0.08, color="red")
    ax1.set_ylabel("USD/TND")
    ax1.set_title("Predicted Intrinsic Value vs Actual IB Rate")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # (b) Prediction errors over time
    ax2 = fig.add_subplot(gs[1, 0])
    errors = preds["intrinsic"] - preds["ib_rate"]
    ax2.plot(preds.index, errors, color=COLORS["spread"], lw=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.fill_between(preds.index, errors, 0,
                     where=errors > 0, color=COLORS["intrinsic"], alpha=0.25)
    ax2.fill_between(preds.index, errors, 0,
                     where=errors < 0, color=COLORS["baseline"], alpha=0.25)
    ax2.set_title("Prediction Errors Over Time")
    ax2.set_ylabel("Error (TND)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # (c) Error distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(errors.values, bins=50, color=COLORS["baseline"],
             alpha=0.7, edgecolor="white", density=True)
    ax3.axvline(0, color="black", lw=1, ls="--")
    from scipy.stats import norm
    xs = np.linspace(errors.min(), errors.max(), 200)
    ax3.plot(xs, norm.pdf(xs, errors.mean(), errors.std()),
             color=COLORS["intrinsic"], lw=2, label="Normal fit")
    stats_text = (f"RMSE={m['RMSE']:.4f}\n"
                  f"MAE={m['MAE']:.4f}\n"
                  f"Dir.acc={m['directional_acc']:.1%}")
    ax3.text(0.05, 0.92, stats_text, transform=ax3.transAxes, fontsize=9,
             va="top", bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax3.set_title("Prediction Error Distribution")
    ax3.set_xlabel("Error (TND)")
    ax3.legend()

    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    metrics = ["MAE", "RMSE", "directional_acc"]
    labels  = comparison_df.index.tolist()
    n       = len(labels)
    colors  = [COLORS["fixing"], COLORS["ib_rate"],
                COLORS["intrinsic"], COLORS["baseline"]][:n]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    fig.suptitle("Model Comparison – Out-of-Sample Metrics", fontsize=14)

    for ax, metric in zip(axes, metrics):
        vals = comparison_df[metric].values
        bars = ax.barh(labels, vals, color=colors[:len(labels)], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}" if metric != "directional_acc" else f"{val:.1%}",
                    va="center", fontsize=9)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        if metric == "directional_acc":
            ax.axvline(0.5, color="red", ls="--", lw=0.8, label="Random (50%)")
            ax.legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Convenience: save all charts
# ---------------------------------------------------------------------------

def save_all_charts(
    data:          pd.DataFrame,
    basket_model,
    kalman_model,
    ou_model       = None,
    backtest_result= None,
    comparison_df  = None,
    output_dir:    str = "output",
):
    os.makedirs(output_dir, exist_ok=True)

    print("[viz] Generating basket diagnostics …")
    plot_basket_diagnostics(basket_model, data,
                             save_path=f"{output_dir}/01_basket_diagnostics.png")

    print("[viz] Generating spread analysis …")
    plot_spread_analysis(data, ou_model,
                          save_path=f"{output_dir}/02_spread_analysis.png")

    print("[viz] Generating Kalman filter chart …")
    plot_kalman_filter(data, kalman_model,
                        save_path=f"{output_dir}/03_kalman_filter.png")

    print("[viz] Generating intrinsic value chart …")
    plot_intrinsic_value(data, save_path=f"{output_dir}/04_intrinsic_value.png")

    if backtest_result is not None:
        print("[viz] Generating backtest performance chart …")
        plot_backtest_performance(backtest_result,
                                   save_path=f"{output_dir}/05_backtest_performance.png")

    if comparison_df is not None:
        print("[viz] Generating model comparison chart …")
        plot_model_comparison(comparison_df,
                               save_path=f"{output_dir}/06_model_comparison.png")

    print(f"[viz] All charts saved to ./{output_dir}/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_stress_periods(preds: pd.DataFrame):
    """Extract contiguous date ranges where stress_regime == True."""
    stress = preds["stress_regime"].fillna(False)
    periods = []
    in_stress = False
    start = None
    for dt, val in stress.items():
        if val and not in_stress:
            start     = dt
            in_stress = True
        elif not val and in_stress:
            periods.append((start, dt))
            in_stress = False
    if in_stress:
        periods.append((start, preds.index[-1]))
    return periods


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  → saved: {save_path}")
    else:
        plt.show()
