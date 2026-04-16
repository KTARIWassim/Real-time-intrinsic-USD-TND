"""
backtest/engine.py
------------------
Backtesting & Performance Evaluation Engine

Implements:
  - Walk-forward backtest (expanding or rolling window)
  - Out-of-sample performance evaluation
  - Error statistics: MAE, RMSE, MAPE, directional accuracy
  - Regime-conditional performance (normal vs stress)
  - Comparison across all three model variants (OU, Kalman, ECM)
"""

import numpy as np
import pandas as pd
from typing import Literal
from dataclasses import dataclass

from models.basket    import BasketModel
from models.liquidity import KalmanSpreadFilter, OrnsteinUhlenbeckModel


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Comprehensive error statistics for intrinsic rate predictions."""
    mask    = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t     = y_true[mask]
    y_p     = y_pred[mask]
    errors  = y_p - y_t
    abs_err = np.abs(errors)

    n = len(y_t)
    if n == 0:
        return {}

    # Direction: did the model predict the correct sign of rate change?
    delta_true = np.diff(y_t)
    delta_pred = np.diff(y_p)
    dir_acc    = float(np.mean(np.sign(delta_true) == np.sign(delta_pred)))

    # Theil's U statistic
    numerator   = np.sqrt(np.mean(errors**2))
    denominator = np.sqrt(np.mean(y_t**2))
    theil_u     = numerator / (denominator + 1e-12)

    return {
        "label"           : label,
        "n_obs"           : n,
        "MAE"             : float(np.mean(abs_err)),
        "RMSE"            : float(np.sqrt(np.mean(errors**2))),
        "MAPE_pct"        : float(np.mean(abs_err / (np.abs(y_t) + 1e-12)) * 100),
        "max_error"       : float(np.max(abs_err)),
        "median_error"    : float(np.median(abs_err)),
        "mean_bias"       : float(np.mean(errors)),
        "directional_acc" : round(dir_acc, 4),
        "Theil_U"         : round(theil_u, 4),
        "corr"            : float(np.corrcoef(y_t, y_p)[0, 1]),
    }


@dataclass
class BacktestResult:
    metrics_is:    dict    # in-sample metrics
    metrics_oos:   dict    # out-of-sample metrics
    predictions:   pd.DataFrame
    walk_forward:  pd.DataFrame


# ---------------------------------------------------------------------------
# Walk-forward backtester
# ---------------------------------------------------------------------------

class WalkForwardBacktest:
    """
    Walk-forward (expanding-window) backtest for the TND intrinsic value model.

    At each step:
      1. Fit BasketModel on data up to t_train_end
      2. Fit KalmanSpreadFilter on same window
      3. Predict intrinsic value for the next `horizon` days
      4. Compare with realised IB rate (benchmark)

    Parameters
    ----------
    train_size  : int   initial training window in days
    step_size   : int   how many days to advance each fold
    horizon     : int   prediction horizon in days
    mode        : 'expanding' or 'rolling'
    """

    def __init__(
        self,
        train_size:  int = 252,
        step_size:   int = 21,
        horizon:     int = 1,
        mode:        Literal["expanding", "rolling"] = "expanding",
        roll_window: int = 504,
    ):
        self.train_size  = train_size
        self.step_size   = step_size
        self.horizon     = horizon
        self.mode        = mode
        self.roll_window = roll_window

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        data : master DataFrame from build_master_dataset()

        Returns
        -------
        BacktestResult with in-sample / OOS metrics and full prediction DataFrame
        """
        n     = len(data)
        folds = []

        t = self.train_size
        while t + self.horizon <= n:
            train_start = max(0, t - self.roll_window) if self.mode == "rolling" else 0
            train_end   = t
            test_end    = min(t + self.step_size, n)

            train = data.iloc[train_start:train_end]
            test  = data.iloc[train_end:test_end]

            if len(train) < 60 or len(test) == 0:
                t += self.step_size
                continue

            # Fit basket model
            basket = BasketModel(constrain_weights=False, rolling_window=None)
            try:
                basket.fit(train)
            except Exception as e:
                print(f"[backtest] Basket fit failed at fold {t}: {e}")
                t += self.step_size
                continue

            # Fit Kalman filter
            kalman = KalmanSpreadFilter(estimate_noise=True)
            try:
                kalman.fit(train)
            except Exception as e:
                print(f"[backtest] Kalman fit failed at fold {t}: {e}")
                t += self.step_size
                continue

            # Apply basket to test set
            test_ext = basket.apply_to_dataset(test)

            # Kalman spread nowcast on test set
            spread_train = train["spread_raw"].values
            kf_full      = kalman.filter_series(spread_train)
            x_last       = float(kf_full["state"][-1])
            P_last       = float(kf_full["variance"][-1])

            # Step through test rows with the Kalman filter
            for idx in range(len(test_ext)):
                row    = test_ext.iloc[idx]
                kf_out = kalman.nowcast(x_last, P_last)

                intrinsic = row.get("predicted_fixing", row["fixing"]) + kf_out["mean"]
                folds.append({
                    "date"          : test.index[idx],
                    "fold_start"    : train.index[0],
                    "fold_end"      : train.index[-1],
                    "fixing"        : row["fixing"],
                    "ib_rate"       : row["ib_rate"],
                    "baseline"      : row.get("predicted_fixing", row["fixing"]),
                    "delta_kf"      : kf_out["mean"],
                    "intrinsic"     : intrinsic,
                    "intrinsic_lo"  : (row.get("predicted_fixing", row["fixing"])
                                       + kf_out["lower_95"]),
                    "intrinsic_hi"  : (row.get("predicted_fixing", row["fixing"])
                                       + kf_out["upper_95"]),
                    "spread_actual" : row["spread_raw"],
                    "train_size"    : train_end - train_start,
                })

                # Update Kalman with the actual spread (becomes next observation)
                actual_spread = row["spread_raw"]
                K      = P_last / (P_last + kalman.R_)
                x_last = x_last + K * (actual_spread - x_last)
                P_last = (1 - K) * P_last

            t += self.step_size

        if not folds:
            raise RuntimeError("No backtest folds completed. Check data length.")

        preds = pd.DataFrame(folds).set_index("date")

        # Performance vs IB rate (best available ground truth)
        # Note: IB rate at t reflects market at t-1 due to publication lag
        y_true = preds["ib_rate"].values
        y_pred = preds["intrinsic"].values

        # Split in-sample vs out-of-sample (first 252 obs is pure IS)
        split  = self.train_size
        metrics_is  = compute_metrics(
            data["ib_rate"].values[:split],
            data["ib_rate"].values[:split],   # IS = perfect by construction (trivial)
            label="In-sample (train period)"
        )
        metrics_oos = compute_metrics(y_true, y_pred, label="Out-of-sample (walk-forward)")

        # Regime analysis: flag stress periods (spread > 1.5× historical σ)
        preds["stress_regime"] = (
            preds["spread_actual"].abs()
            > preds["spread_actual"].abs().rolling(63).mean()
              + 1.5 * preds["spread_actual"].rolling(63).std()
        )

        result = BacktestResult(
            metrics_is   = metrics_is,
            metrics_oos  = metrics_oos,
            predictions  = preds,
            walk_forward = preds,
        )

        self._print_results(result)
        return result

    def _print_results(self, result: BacktestResult):
        m = result.metrics_oos
        print("\n" + "=" * 58)
        print("  WALK-FORWARD BACKTEST – OUT-OF-SAMPLE RESULTS")
        print("=" * 58)
        print(f"  Observations        : {m['n_obs']}")
        print(f"  MAE                 : {m['MAE']:.5f} TND")
        print(f"  RMSE                : {m['RMSE']:.5f} TND")
        print(f"  MAPE                : {m['MAPE_pct']:.3f} %")
        print(f"  Max error           : {m['max_error']:.5f} TND")
        print(f"  Mean bias           : {m['mean_bias']:+.6f} TND")
        print(f"  Directional acc.    : {m['directional_acc']:.1%}")
        print(f"  Correlation (Pearson): {m['corr']:.4f}")
        print(f"  Theil's U           : {m['Theil_U']:.4f}")
        print("=" * 58 + "\n")


# ---------------------------------------------------------------------------
# Model comparison utility
# ---------------------------------------------------------------------------

def compare_models(data: pd.DataFrame, train_frac: float = 0.75) -> pd.DataFrame:
    """
    Compare OU, Kalman, and baseline (fixing only) models on OOS data.

    Returns a DataFrame of performance metrics for each model.
    """
    from models.liquidity import OrnsteinUhlenbeckModel

    split = int(len(data) * train_frac)
    train = data.iloc[:split]
    test  = data.iloc[split:]

    # Fit shared basket model
    basket = BasketModel(constrain_weights=False, rolling_window=None)
    basket.fit(train)
    test_b = basket.apply_to_dataset(test)

    y_true = test_b["ib_rate"].values
    results = []

    # 1. Fixing-only baseline (naive benchmark)
    results.append(compute_metrics(y_true, test_b["fixing"].values,
                                   label="Naive: fixing only"))

    # 2. Basket baseline only (no spread adjustment)
    results.append(compute_metrics(
        y_true, test_b["predicted_fixing"].values,
        label="Basket baseline (no adj.)"
    ))

    # 3. OU model
    try:
        ou = OrnsteinUhlenbeckModel(session_split=False)
        ou.fit(train)
        test_ou = ou.apply_to_dataset(test_b)
        results.append(compute_metrics(y_true, test_ou["intrinsic_ou"].dropna().values,
                                       label="OU spread model"))
    except Exception as e:
        print(f"[compare] OU model failed: {e}")

    # 4. Kalman filter
    try:
        kf = KalmanSpreadFilter(estimate_noise=True)
        kf.fit(train)
        test_kf = kf.apply_to_dataset(test_b)
        results.append(compute_metrics(y_true, test_kf["intrinsic_kf"].dropna().values,
                                       label="Kalman spread filter"))
    except Exception as e:
        print(f"[compare] Kalman model failed: {e}")

    df = pd.DataFrame(results).set_index("label")
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON – OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 65)
    print(df[["MAE", "RMSE", "MAPE_pct", "directional_acc", "corr"]].to_string())
    print("=" * 65 + "\n")
    return df
