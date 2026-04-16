"""
models/basket.py
----------------
Component 1 – Basket-Based Baseline Model

Estimates the weights (w1, w2, w3) that link movements in global FX pairs
to movements in the BCT USD/TND fixing rate, then uses those weights to
compute a continuously updating intrinsic baseline value intraday.

Model
-----
    Δln(fixing_t) = w1·Δln(EURUSD_t) + w2·Δln(GBPUSD_t) + w3·Δln(USDJPY_t) + ε_t

Estimated via OLS (and optionally constrained OLS).
The baseline rate at any intraday moment t is then:

    baseline(t) = fixing_anchor × exp( w1·Δln(EURUSD_t) + w2·Δln(GBPUSD_t) + w3·Δln(USDJPY_t) )

where fixing_anchor is the last BCT published fixing.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BasketModelResult:
    weights: dict          # {EURUSD: w1, GBPUSD: w2, USDJPY: w3}
    intercept: float
    r_squared: float
    adj_r_squared: float
    residuals: pd.Series
    fitted_values: pd.Series
    ols_summary: object    # statsmodels RegressionResultsWrapper
    constrained: bool = False
    rolling_weights: Optional[pd.DataFrame] = field(default=None)


# ---------------------------------------------------------------------------
# Basket Model
# ---------------------------------------------------------------------------

class BasketModel:
    """
    Estimates and applies the basket-based USD/TND baseline model.

    Parameters
    ----------
    constrain_weights : bool
        If True, enforce w1 + w2 + w3 = 1 via scipy optimize.
        If False, use unconstrained OLS (recommended to start).
    rolling_window : int or None
        If set, also compute rolling OLS with this window size (days)
        to assess parameter stability over time.
    """

    FEATURE_COLS = ["log_ret_EURUSD", "log_ret_GBPUSD", "log_ret_USDJPY"]
    TARGET_COL   = "log_ret_fixing"

    def __init__(self, constrain_weights: bool = False, rolling_window: int = 90):
        self.constrain_weights = constrain_weights
        self.rolling_window    = rolling_window
        self.result_: Optional[BasketModelResult] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "BasketModel":
        """
        Fit the basket regression on the master dataset.

        Parameters
        ----------
        data : DataFrame from data.loader.build_master_dataset()
        """
        df = data[self.FEATURE_COLS + [self.TARGET_COL]].dropna().copy()

        X = df[self.FEATURE_COLS].values
        y = df[self.TARGET_COL].values

        if self.constrain_weights:
            result = self._fit_constrained(X, y, df)
        else:
            result = self._fit_ols(X, y, df)

        # Rolling weights for stability analysis
        rolling_w = None
        if self.rolling_window:
            rolling_w = self._rolling_ols(df)

        self.result_ = BasketModelResult(
            weights       = result["weights"],
            intercept     = result["intercept"],
            r_squared     = result["r2"],
            adj_r_squared = result["adj_r2"],
            residuals     = result["residuals"],
            fitted_values = result["fitted"],
            ols_summary   = result.get("summary"),
            constrained   = self.constrain_weights,
            rolling_weights = rolling_w,
        )
        self._print_summary()
        return self

    def _fit_ols(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> dict:
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        weights = dict(zip(self.FEATURE_COLS, model.params[1:]))
        fitted  = pd.Series(model.fittedvalues, index=df.index)
        resid   = pd.Series(model.resid,        index=df.index)

        return {
            "weights"   : weights,
            "intercept" : float(model.params[0]),
            "r2"        : float(model.rsquared),
            "adj_r2"    : float(model.rsquared_adj),
            "residuals" : resid,
            "fitted"    : fitted,
            "summary"   : model,
        }

    def _fit_constrained(self, X: np.ndarray, y: np.ndarray,
                          df: pd.DataFrame) -> dict:
        """Constrained OLS: w1+w2+w3=1, no intercept."""
        def loss(w):
            pred = X @ w
            return np.sum((y - pred) ** 2)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds      = [(-2, 3)] * 3            # allow negative but bounded
        w0          = np.array([0.6, 0.2, 0.2])

        opt = minimize(loss, w0, method="SLSQP",
                       constraints=constraints, bounds=bounds,
                       options={"ftol": 1e-10, "maxiter": 1000})

        if not opt.success:
            raise RuntimeError(f"Constrained optimisation failed: {opt.message}")

        weights = dict(zip(self.FEATURE_COLS, opt.x))
        pred    = X @ opt.x
        ss_res  = np.sum((y - pred) ** 2)
        ss_tot  = np.sum((y - y.mean()) ** 2)
        r2      = 1.0 - ss_res / ss_tot
        n, p    = len(y), 3
        adj_r2  = 1.0 - (1 - r2) * (n - 1) / (n - p - 1)

        fitted = pd.Series(pred, index=df.index)
        resid  = pd.Series(y - pred, index=df.index)

        return {
            "weights"   : weights,
            "intercept" : 0.0,
            "r2"        : r2,
            "adj_r2"    : adj_r2,
            "residuals" : resid,
            "fitted"    : fitted,
            "summary"   : None,
        }

    def _rolling_ols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling OLS weights to test parameter stability."""
        w = self.rolling_window
        records = []
        for i in range(w, len(df) + 1):
            chunk = df.iloc[i - w : i]
            X_sm  = sm.add_constant(chunk[self.FEATURE_COLS].values)
            y_sub = chunk[self.TARGET_COL].values
            try:
                m    = sm.OLS(y_sub, X_sm).fit()
                row  = {col: m.params[j+1]
                        for j, col in enumerate(self.FEATURE_COLS)}
                row["date"] = df.index[i - 1]
                records.append(row)
            except Exception:
                pass
        if not records:
            return pd.DataFrame()
        rw = pd.DataFrame(records).set_index("date")
        # Rename columns to clean labels
        rw.columns = ["w_EURUSD", "w_GBPUSD", "w_USDJPY"]
        return rw

    # ------------------------------------------------------------------
    # Prediction / real-time update
    # ------------------------------------------------------------------

    def predict_log_return(self, delta_log_fx: dict) -> float:
        """
        Predict the log-return of the fixing for a given set of FX log-returns.

        Parameters
        ----------
        delta_log_fx : dict  e.g. {'log_ret_EURUSD': 0.001, ...}

        Returns
        -------
        float : predicted Δln(fixing)
        """
        self._check_fitted()
        result = self.result_
        pred   = result.intercept
        for col, w in result.weights.items():
            pred += w * delta_log_fx.get(col, 0.0)
        return pred

    def compute_baseline(
        self,
        fixing_anchor: float,
        current_fx: pd.Series,
        anchor_fx: pd.Series,
    ) -> float:
        """
        Compute the real-time baseline USD/TND rate.

        baseline(t) = fixing_anchor × exp( Σ wᵢ · Δln(FXᵢ) )

        Parameters
        ----------
        fixing_anchor : float  last published BCT fixing
        current_fx    : Series {'EURUSD': x, 'GBPUSD': y, 'USDJPY': z}
        anchor_fx     : Series  FX levels at the time of the fixing anchor
        """
        self._check_fitted()
        cum_move = 0.0
        pairs    = {"EURUSD": "log_ret_EURUSD",
                    "GBPUSD": "log_ret_GBPUSD",
                    "USDJPY": "log_ret_USDJPY"}
        for name, feat in pairs.items():
            if anchor_fx[name] > 0:
                delta_log = np.log(current_fx[name] / anchor_fx[name])
                cum_move  += self.result_.weights[feat] * delta_log
        return fixing_anchor * np.exp(cum_move)

    def apply_to_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted model to a full dataset, generating baseline estimates.
        Reconstructs the fixing path from log-return predictions.

        Returns a copy of data with added columns:
            predicted_log_ret, predicted_fixing, basket_residual
        """
        self._check_fitted()
        df = data.copy()

        # Predicted log-returns
        df["predicted_log_ret"] = (
            self.result_.intercept
            + sum(
                self.result_.weights[c] * df[c]
                for c in self.FEATURE_COLS
            )
        )

        # Reconstruct price level from cumulative log-returns
        # Start from the first actual fixing value
        log_fixing_0   = np.log(df["fixing"].iloc[0])
        cum_pred_log   = df["predicted_log_ret"].cumsum()
        df["predicted_fixing"] = np.exp(log_fixing_0 + cum_pred_log)

        df["basket_residual"] = df["fixing"] - df["predicted_fixing"]
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        """Return a dict of diagnostic statistics."""
        self._check_fitted()
        r  = self.result_
        w  = r.weights

        stats = {
            "R_squared"    : round(r.r_squared, 4),
            "Adj_R_squared": round(r.adj_r_squared, 4),
            "weights"      : {k: round(v, 6) for k, v in w.items()},
            "intercept"    : round(r.intercept, 8),
            "weight_sum"   : round(sum(w.values()), 6),
            "residual_std" : round(float(r.residuals.std()), 6),
            "residual_mean": round(float(r.residuals.mean()), 8),
        }
        # Durbin-Watson
        resid = r.residuals.values
        dw    = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
        stats["Durbin_Watson"] = round(dw, 4)
        return stats

    def _print_summary(self):
        d = self.diagnostics()
        print("\n" + "=" * 55)
        print("  BASKET MODEL – FIT SUMMARY")
        print("=" * 55)
        print(f"  R²           : {d['R_squared']}")
        print(f"  Adj R²       : {d['Adj_R_squared']}")
        print(f"  Durbin-Watson: {d['Durbin_Watson']}")
        print(f"  Residual std : {d['residual_std']:.6f}")
        print(f"  Weights:")
        for name, w in d["weights"].items():
            label = name.replace("log_ret_", "")
            print(f"    {label:<12}: {w:+.6f}")
        print(f"  Weight sum   : {d['weight_sum']}")
        print("=" * 55 + "\n")

    def _check_fitted(self):
        if self.result_ is None:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")
