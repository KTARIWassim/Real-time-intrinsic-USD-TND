"""
models/realtime.py
------------------
Real-Time Intrinsic Value Engine

Combines Component 1 (basket baseline) and Component 2 (Kalman liquidity
adjustment) into a continuously updating intrinsic USD/TND estimate.

    V*(t) = f(fixing_anchor, ΔFX_t) + δ̂(t)

The engine is designed to:
  - Accept streaming FX ticks (simulated or live)
  - Update the baseline instantly on each tick
  - Update the Kalman spread estimate when new information arrives
  - Output a complete state dict at each update

For live deployment, replace the simulated tick generator with a WebSocket
or REST polling feed from your data provider.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Generator
from dataclasses import dataclass, field

from models.basket    import BasketModel
from models.liquidity import KalmanSpreadFilter


# ---------------------------------------------------------------------------
# State object
# ---------------------------------------------------------------------------

@dataclass
class IntrinsicValueState:
    timestamp:       datetime
    fixing_anchor:   float       # last BCT published fixing
    baseline:        float       # basket-implied value
    delta_mean:      float       # Kalman spread nowcast (mean)
    delta_std:       float       # Kalman spread uncertainty
    intrinsic:       float       # V* = baseline + delta_mean
    intrinsic_lo:    float       # lower 95% CI
    intrinsic_hi:    float       # upper 95% CI
    last_ib_rate:    float       # last published IB rate (t-1)
    current_fx:      dict = field(default_factory=dict)
    session:         str  = "morning"


# ---------------------------------------------------------------------------
# Real-time engine
# ---------------------------------------------------------------------------

class RealTimeEngine:
    """
    Orchestrates real-time intrinsic TND valuation.

    Parameters
    ----------
    basket_model  : fitted BasketModel
    kalman_model  : fitted KalmanSpreadFilter
    fixing_anchor : float   – most recent BCT fixing rate
    anchor_fx     : dict    – FX levels at time of fixing anchor
    last_ib_rate  : float   – most recently published IB rate
    init_state    : float   – initial Kalman state (spread estimate)
    init_variance : float   – initial Kalman variance
    """

    def __init__(
        self,
        basket_model:  BasketModel,
        kalman_model:  KalmanSpreadFilter,
        fixing_anchor: float,
        anchor_fx:     dict,
        last_ib_rate:  float,
        init_state:    float = None,
        init_variance: float = None,
    ):
        self.basket  = basket_model
        self.kalman  = kalman_model

        self._fixing_anchor = fixing_anchor
        self._anchor_fx     = anchor_fx
        self._last_ib       = last_ib_rate

        # Kalman state
        self._x = init_state    if init_state    is not None else last_ib_rate - fixing_anchor
        self._P = init_variance if init_variance is not None else kalman_model.Q_

        self._history: list[IntrinsicValueState] = []

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        current_fx:      dict,
        timestamp:       datetime = None,
        intraday_signal: Optional[float] = None,
        signal_noise:    Optional[float] = None,
        session:         str = "morning",
    ) -> IntrinsicValueState:
        """
        Process a new FX tick and return the updated intrinsic value state.

        Parameters
        ----------
        current_fx      : dict  {'EURUSD': float, 'GBPUSD': float, 'USDJPY': float}
        intraday_signal : float or None  – optional indicative IB quote (partial signal)
        signal_noise    : float or None  – std of the intraday signal noise
        session         : 'morning' | 'afternoon'
        """
        ts = timestamp or datetime.utcnow()

        # 1. Basket baseline
        anchor_series  = pd.Series(self._anchor_fx)
        current_series = pd.Series(current_fx)
        baseline = self.basket.compute_baseline(
            self._fixing_anchor, current_series, anchor_series
        )

        # 2. Kalman nowcast
        obs_noise = signal_noise if signal_noise is not None else self.kalman.R_
        kf_out    = self.kalman.nowcast(
            x_prev      = self._x,
            P_prev      = self._P,
            observation = intraday_signal,
            obs_noise   = obs_noise,
        )

        # Update Kalman state (store posterior for next tick)
        self._x = kf_out["mean"]
        self._P = kf_out["variance"]

        # 3. Combine
        intrinsic    = baseline + kf_out["mean"]
        intrinsic_lo = baseline + kf_out["lower_95"]
        intrinsic_hi = baseline + kf_out["upper_95"]

        state = IntrinsicValueState(
            timestamp     = ts,
            fixing_anchor = self._fixing_anchor,
            baseline      = baseline,
            delta_mean    = kf_out["mean"],
            delta_std     = kf_out["std"],
            intrinsic     = intrinsic,
            intrinsic_lo  = intrinsic_lo,
            intrinsic_hi  = intrinsic_hi,
            last_ib_rate  = self._last_ib,
            current_fx    = current_fx.copy(),
            session       = session,
        )

        self._history.append(state)
        return state

    # ------------------------------------------------------------------
    # Fixing anchor update (called twice daily when BCT publishes)
    # ------------------------------------------------------------------

    def update_fixing(
        self,
        new_fixing:   float,
        new_anchor_fx: dict,
        new_ib_rate:  Optional[float] = None,
    ):
        """
        Reset the baseline anchor when a new BCT fixing is published.
        Optionally incorporate the new IB rate (t-1 observation) into the
        Kalman filter.
        """
        print(f"[engine] Fixing anchor updated: {self._fixing_anchor:.4f} → {new_fixing:.4f}")
        self._fixing_anchor = new_fixing
        self._anchor_fx     = new_anchor_fx

        if new_ib_rate is not None:
            # Use the new IB rate as an observation to update the Kalman state
            new_spread = new_ib_rate - new_fixing
            obs_noise  = self.kalman.R_
            K          = self._P / (self._P + obs_noise)
            self._x    = self._x + K * (new_spread - self._x)
            self._P    = (1 - K) * self._P
            self._last_ib = new_ib_rate
            print(f"[engine] Kalman updated with IB rate {new_ib_rate:.4f}, "
                  f"spread nowcast = {self._x:.5f}")

    # ------------------------------------------------------------------
    # Simulation / backtesting feed
    # ------------------------------------------------------------------

    def stream_historical(
        self,
        data:       pd.DataFrame,
        tick_every: int = 1,
    ) -> Generator[IntrinsicValueState, None, None]:
        """
        Replay historical daily data through the engine, yielding a state
        at each row.  In production, replace this with live tick ingestion.

        Parameters
        ----------
        data       : master DataFrame (from build_master_dataset)
        tick_every : process every N rows (for speed when replaying dense data)
        """
        for i, (ts, row) in enumerate(data.iterrows()):
            if i % tick_every != 0:
                continue

            current_fx = {
                "EURUSD": row["EURUSD"],
                "GBPUSD": row["GBPUSD"],
                "USDJPY": row["USDJPY"],
            }

            state = self.update(
                current_fx  = current_fx,
                timestamp   = pd.Timestamp(ts).to_pydatetime(),
                session     = "morning",
            )
            yield state

    def history_dataframe(self) -> pd.DataFrame:
        """Convert the internal history list to a tidy DataFrame."""
        if not self._history:
            return pd.DataFrame()
        records = []
        for s in self._history:
            records.append({
                "timestamp"    : s.timestamp,
                "fixing_anchor": s.fixing_anchor,
                "baseline"     : s.baseline,
                "delta_mean"   : s.delta_mean,
                "delta_std"    : s.delta_std,
                "intrinsic"    : s.intrinsic,
                "intrinsic_lo" : s.intrinsic_lo,
                "intrinsic_hi" : s.intrinsic_hi,
                "last_ib_rate" : s.last_ib_rate,
                "EURUSD"       : s.current_fx.get("EURUSD"),
                "GBPUSD"       : s.current_fx.get("GBPUSD"),
                "USDJPY"       : s.current_fx.get("USDJPY"),
            })
        df = pd.DataFrame(records).set_index("timestamp")
        return df
