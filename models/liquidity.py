"""
models/liquidity.py
-------------------
Component 2 – Stochastic Liquidity Adjustment  δ(t)

Models and nowcasts the spread between the BCT interbank rate and the
official fixing, capturing local FX market frictions in real time.

Three implementations are provided:

  1. OrnsteinUhlenbeckModel   – parametric mean-reversion (analytical MLE)
  2. KalmanSpreadFilter       – state-space nowcaster (handles the t-1 lag)
  3. ErrorCorrectionModel     – cointegration-based ECM via VECM

All models expose the same interface:
    .fit(data)  →  self
    .nowcast(prev_spread, intraday_signals)  →  float
    .apply_to_dataset(data)  →  DataFrame with delta_estimate column
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm as scipy_norm
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Ornstein-Uhlenbeck  (closed-form MLE)
# ---------------------------------------------------------------------------

@dataclass
class OUParams:
    theta: float   # mean-reversion speed  (per day)
    mu:    float   # long-run mean spread
    sigma: float   # diffusion (daily vol)


class OrnsteinUhlenbeckModel:
    """
    Continuous-time OU process for the spread S(t):

        dS = θ(μ - S)dt + σ dW

    Discretised (exact) for daily data:
        S_t = μ + e^{-θΔ}(S_{t-1} - μ) + σ√((1-e^{-2θΔ})/(2θ)) · ε_t

    Parameters estimated via maximum likelihood.
    """

    def __init__(self, session_split: bool = True):
        self.session_split = session_split   # fit AM / PM regimes separately
        self.params_: Optional[OUParams] = None
        self.params_am_: Optional[OUParams] = None
        self.params_pm_: Optional[OUParams] = None
        self._dt = 1.0   # daily step

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "OrnsteinUhlenbeckModel":
        s = data["spread_raw"].dropna()
        self.params_ = self._mle(s)

        if self.session_split:
            # Proxy AM = first half of year, PM = second half (or use actual
            # intraday timestamp if available)
            mid   = len(s) // 2
            s_am  = s.iloc[:mid]
            s_pm  = s.iloc[mid:]
            self.params_am_ = self._mle(s_am) if len(s_am) > 30 else self.params_
            self.params_pm_ = self._mle(s_pm) if len(s_pm) > 30 else self.params_

        self._print_params()
        return self

    def _mle(self, s: pd.Series) -> OUParams:
        """Exact MLE for OU parameters given discrete observations."""
        x  = s.values
        n  = len(x)
        Δ  = self._dt

        def neg_loglik(params):
            θ, μ, σ = params
            if θ <= 0 or σ <= 0:
                return 1e12
            e      = np.exp(-θ * Δ)
            σ_sq_e = σ**2 * (1 - np.exp(-2*θ*Δ)) / (2*θ)
            if σ_sq_e <= 0:
                return 1e12
            σ_e  = np.sqrt(σ_sq_e)
            resid = x[1:] - (μ + e * (x[:-1] - μ))
            ll    = -0.5 * n * np.log(2 * np.pi * σ_sq_e) \
                    - 0.5 * np.sum(resid**2) / σ_sq_e
            return -ll

        # Method-of-moments starting values
        mu0    = x.mean()
        theta0 = 0.3
        sigma0 = x.std() * np.sqrt(2 * theta0)

        result = minimize(
            neg_loglik,
            x0     = [theta0, mu0, sigma0],
            method = "L-BFGS-B",
            bounds  = [(1e-4, 20), (-0.1, 0.1), (1e-6, 0.1)],
            options = {"ftol": 1e-10, "maxiter": 2000},
        )
        if not result.success:
            warnings.warn(f"OU MLE did not fully converge: {result.message}")

        θ, μ, σ = result.x
        return OUParams(theta=θ, mu=μ, sigma=σ)

    # ------------------------------------------------------------------
    # Nowcasting
    # ------------------------------------------------------------------

    def nowcast(
        self,
        prev_spread: float,
        session:     str = "morning",
        horizon:     float = 1.0,
    ) -> dict:
        """
        Nowcast the spread h steps ahead (h=1 → next-day IB rate).

        Returns dict with 'mean', 'std', 'lower_95', 'upper_95'.
        """
        p   = (self.params_am_ if session == "morning" else self.params_pm_) \
               or self.params_
        e   = np.exp(-p.theta * horizon * self._dt)
        μ_h = p.mu + e * (prev_spread - p.mu)
        σ_h = p.sigma * np.sqrt((1 - np.exp(-2*p.theta*horizon*self._dt))
                                 / (2*p.theta))
        return {
            "mean"    : μ_h,
            "std"     : σ_h,
            "lower_95": μ_h - 1.96 * σ_h,
            "upper_95": μ_h + 1.96 * σ_h,
        }

    def apply_to_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # One-step-ahead OU forecast of the spread (nowcast for same day)
        spread = df["spread_raw"].values
        p      = self.params_
        e      = np.exp(-p.theta * self._dt)
        forecast = p.mu + e * (spread - p.mu)
        df["delta_estimate_ou"] = np.concatenate([[np.nan], forecast[:-1]])
        df["intrinsic_ou"]      = df["predicted_fixing"] + df["delta_estimate_ou"]
        return df

    def _print_params(self):
        p = self.params_
        print("\n" + "=" * 50)
        print("  ORNSTEIN-UHLENBECK SPREAD MODEL")
        print("=" * 50)
        print(f"  θ (reversion speed) : {p.theta:.4f}  "
              f"(half-life ≈ {np.log(2)/p.theta:.1f} days)")
        print(f"  μ (long-run mean)   : {p.mu:.6f} TND")
        print(f"  σ (diffusion)       : {p.sigma:.6f} TND/√day")
        if self.session_split and self.params_am_:
            print(f"  AM regime σ         : {self.params_am_.sigma:.6f}")
            print(f"  PM regime σ         : {self.params_pm_.sigma:.6f}")
        print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# 2. Kalman Filter Spread Nowcaster
# ---------------------------------------------------------------------------

class KalmanSpreadFilter:
    """
    State-space model for the latent true spread S*(t).

    State equation (transition):
        S*(t) = φ·S*(t-1) + η_t,    η_t ~ N(0, Q)

    Observation equation:
        S_obs(t-1) = S*(t-1) + ε_t,  ε_t ~ N(0, R)

    Because the IB rate is published with a 1-day lag, the observation
    at time t is actually an observation of the state at t-1.  The filter
    handles this via a modified update step.

    At each intraday point, partial signals (e.g. indicative quotes) can
    be used to update the posterior via standard Kalman update equations.
    """

    def __init__(self, estimate_noise: bool = True):
        self.estimate_noise = estimate_noise
        self.phi_:   Optional[float] = None    # state transition
        self.Q_:     Optional[float] = None    # process noise variance
        self.R_:     Optional[float] = None    # observation noise variance
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting (EM or method of moments)
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "KalmanSpreadFilter":
        s = data["spread_raw"].dropna().values

        if self.estimate_noise:
            self.phi_, self.Q_, self.R_ = self._em_estimate(s)
        else:
            # Method-of-moments defaults
            self.phi_ = 0.85
            self.Q_   = np.var(np.diff(s)) / 2
            self.R_   = np.var(s) * 0.05

        self._fitted = True
        self._print_params(s)
        return self

    def _em_estimate(self, s: np.ndarray, n_iter: int = 50) -> tuple:
        """
        Expectation-Maximisation to jointly estimate φ, Q, R.
        Uses the Kalman smoother (RTS smoother) in the E-step.
        """
        n   = len(s)
        phi = 0.85
        Q   = float(np.var(np.diff(s))) / 2
        R   = float(np.var(s)) * 0.05

        for _ in range(n_iter):
            # --- E step: Kalman filter + smoother ---
            x_f = np.zeros(n)    # filtered state
            P_f = np.zeros(n)    # filtered variance
            x_p = np.zeros(n)    # predicted state
            P_p = np.zeros(n)    # predicted variance

            x_p[0] = s[0]
            P_p[0] = Q / (1 - phi**2 + 1e-10)

            for t in range(n):
                # Update (observation at t-1 informs state at t-1, not t)
                K      = P_p[t] / (P_p[t] + R)
                x_f[t] = x_p[t] + K * (s[t] - x_p[t])
                P_f[t] = (1 - K) * P_p[t]
                # Predict
                if t < n - 1:
                    x_p[t+1] = phi * x_f[t]
                    P_p[t+1] = phi**2 * P_f[t] + Q

            # RTS smoother
            x_s = x_f.copy()
            P_s = P_f.copy()
            P_cross = np.zeros(n - 1)   # E[x_t x_{t-1}]
            for t in range(n-2, -1, -1):
                G        = P_f[t] * phi / P_p[t+1]
                x_s[t]   = x_f[t] + G * (x_s[t+1] - x_p[t+1])
                P_s[t]   = P_f[t] + G**2 * (P_s[t+1] - P_p[t+1])
                P_cross[t] = G * P_s[t+1] + x_s[t] * x_s[t+1]

            # --- M step ---
            E_xx  = np.sum(P_s[1:] + x_s[1:]**2)
            E_x1x = np.sum(P_cross + x_s[1:]*x_s[:-1])
            E_x0  = np.sum(P_s[:-1] + x_s[:-1]**2)

            phi = E_x1x / (E_x0 + 1e-10)
            phi = np.clip(phi, -0.999, 0.999)

            Q = (np.sum(P_s[1:] + x_s[1:]**2)
                 - phi * E_x1x
                 - phi * E_x1x
                 + phi**2 * E_x0) / n
            Q = max(Q, 1e-10)

            innov = s - x_s
            R     = float(np.mean(innov**2 + P_s))
            R     = max(R, 1e-10)

        return float(phi), float(Q), float(R)

    # ------------------------------------------------------------------
    # Real-time filtering
    # ------------------------------------------------------------------

    def filter_series(self, spreads: np.ndarray) -> dict:
        """
        Run the Kalman filter over a full spread series.

        Returns dict with arrays: state, variance, innovations, gain
        """
        self._check_fitted()
        n    = len(spreads)
        phi, Q, R = self.phi_, self.Q_, self.R_

        x_f    = np.zeros(n)
        P_f    = np.zeros(n)
        innov  = np.zeros(n)
        K_gain = np.zeros(n)

        # Initialise at unconditional mean/variance
        x_p = spreads[0]
        P_p = Q / (1 - phi**2 + 1e-10)

        for t in range(n):
            # Innovation
            innov[t]  = spreads[t] - x_p
            S_t       = P_p + R
            K_gain[t] = P_p / S_t

            # Update
            x_f[t] = x_p + K_gain[t] * innov[t]
            P_f[t] = (1 - K_gain[t]) * P_p

            # Predict
            x_p = phi * x_f[t]
            P_p = phi**2 * P_f[t] + Q

        # One-step-ahead predictions (the nowcast)
        x_pred = np.concatenate([[spreads[0]], phi * x_f[:-1]])
        P_pred = np.concatenate([[P_f[0]],     phi**2 * P_f[:-1] + Q])

        return {
            "state"    : x_f,
            "variance" : P_f,
            "predicted": x_pred,
            "pred_var" : P_pred,
            "innovations": innov,
            "kalman_gain": K_gain,
        }

    def nowcast(
        self,
        x_prev: float,     # filtered state estimate at t-1
        P_prev: float,     # filtered variance at t-1
        observation: Optional[float] = None,   # partial intraday signal
        obs_noise:   float = None,             # noise of the intraday signal
    ) -> dict:
        """
        One-step nowcast for the current spread, optionally incorporating
        a partial intraday signal (e.g. an indicative bank quote).

        Returns {'mean': float, 'variance': float, 'std': float, ...}
        """
        self._check_fitted()
        phi, Q, R = self.phi_, self.Q_, self.R_

        # Predict
        x_pred = phi * x_prev
        P_pred = phi**2 * P_prev + Q

        if observation is not None:
            # Update with partial intraday signal
            R_obs  = obs_noise if obs_noise is not None else R
            K      = P_pred / (P_pred + R_obs)
            x_upd  = x_pred + K * (observation - x_pred)
            P_upd  = (1 - K) * P_pred
        else:
            x_upd, P_upd = x_pred, P_pred

        std_upd = np.sqrt(P_upd)
        return {
            "mean"    : x_upd,
            "variance": P_upd,
            "std"     : std_upd,
            "lower_95": x_upd - 1.96 * std_upd,
            "upper_95": x_upd + 1.96 * std_upd,
        }

    def apply_to_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Kalman filter to the full dataset and store the spread nowcast."""
        self._check_fitted()
        df     = data.copy()
        spread = df["spread_raw"].values
        kf     = self.filter_series(spread)

        df["delta_estimate_kf"]   = kf["predicted"]   # one-step-ahead Kalman
        df["delta_variance_kf"]   = kf["pred_var"]
        df["delta_upper_95_kf"]   = (kf["predicted"]
                                      + 1.96 * np.sqrt(kf["pred_var"]))
        df["delta_lower_95_kf"]   = (kf["predicted"]
                                      - 1.96 * np.sqrt(kf["pred_var"]))
        df["intrinsic_kf"]        = (df.get("predicted_fixing", df["fixing"])
                                      + df["delta_estimate_kf"])
        return df

    def _print_params(self, s: np.ndarray):
        hl = -np.log(2) / np.log(abs(self.phi_)) if abs(self.phi_) < 1 else np.inf
        print("\n" + "=" * 50)
        print("  KALMAN SPREAD FILTER")
        print("=" * 50)
        print(f"  φ (persistence)     : {self.phi_:.4f}  "
              f"(half-life ≈ {hl:.1f} days)")
        print(f"  Q (process noise)   : {self.Q_:.2e}")
        print(f"  R (obs noise)       : {self.R_:.2e}")
        print(f"  Signal-to-noise     : {self.Q_/self.R_:.4f}")
        print("=" * 50 + "\n")

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# 3. Error-Correction Model (VECM)
# ---------------------------------------------------------------------------

class ErrorCorrectionModel:
    """
    Models the joint dynamics of [IB_rate, fixing] as a VECM, exploiting
    the cointegrating relationship implied by the long-run peg mechanism.

    The ECM term captures the 'pull-back' from deviations, giving a
    structural interpretation for the liquidity spread adjustment.
    """

    def __init__(self, lags: int = 2):
        self.lags   = lags
        self._model = None
        self._result = None
        self._coint_rank = None

    def fit(self, data: pd.DataFrame) -> "ErrorCorrectionModel":
        df = data[["ib_rate", "fixing"]].dropna()

        # Test cointegration rank
        rank_res = select_coint_rank(df.values, det_order=0, k_ar_diff=self.lags,
                                     signif=0.05)
        self._coint_rank = int(rank_res.rank)
        print(f"[ECM] Johansen test: cointegration rank = {self._coint_rank}")

        if self._coint_rank == 0:
            warnings.warn("No cointegration detected – ECM may be misspecified.")
            self._coint_rank = 1   # force r=1 for estimation

        self._model  = VECM(df.values, k_ar_diff=self.lags,
                            coint_rank=self._coint_rank, deterministic="co")
        self._result = self._model.fit()
        print("[ECM] VECM fitted.")
        print(f"  Loading matrix α:\n{self._result.alpha}")
        print(f"  Cointegrating vector β:\n{self._result.beta}")
        return self

    def apply_to_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate in-sample one-step-ahead forecasts of the IB rate.
        The predicted IB minus the fixing gives the ECM spread estimate.
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        df   = data.copy()
        endog = df[["ib_rate", "fixing"]].dropna().values

        n_skip = self.lags + 1
        preds  = []
        for t in range(n_skip, len(endog)):
            fc   = self._result.predict(steps=1)
            preds.append(fc[0, 0])   # IB rate forecast

        pred_series = pd.Series(
            [np.nan] * (len(df) - len(preds)) + preds,
            index=df.index,
        )
        df["delta_estimate_ecm"] = pred_series - df["fixing"]
        df["intrinsic_ecm"]      = (df.get("predicted_fixing", df["fixing"])
                                     + df["delta_estimate_ecm"])
        return df


# ---------------------------------------------------------------------------
# 4. Regime-aware wrapper
# ---------------------------------------------------------------------------

class RegimeAwareAdjustment:
    """
    Wraps any of the three models above with a session (AM/PM) regime switch.

    Detects regime from timestamp (if available) or uses external regime flag.
    Routes nowcast calls to the appropriate session-calibrated model.
    """

    def __init__(self, base_model, regime_col: str = "session"):
        self.model      = base_model
        self.regime_col = regime_col

    def nowcast(self, prev_spread: float, context: dict) -> dict:
        session = context.get("session", "morning")
        if hasattr(self.model, "nowcast"):
            kwargs = {}
            if isinstance(self.model, OrnsteinUhlenbeckModel):
                kwargs["session"] = session
            elif isinstance(self.model, KalmanSpreadFilter):
                kwargs["x_prev"]     = context.get("x_prev", prev_spread)
                kwargs["P_prev"]     = context.get("P_prev", self.model.Q_)
                kwargs["observation"]= context.get("intraday_signal")
                kwargs["obs_noise"]  = context.get("signal_noise")
                return self.model.nowcast(**kwargs)
            return self.model.nowcast(prev_spread, **kwargs)
        raise NotImplementedError
