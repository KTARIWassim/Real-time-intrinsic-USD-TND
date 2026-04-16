"""
main.py
-------
End-to-End Pipeline for the Real-Time Intrinsic TND Valuation Model
FIN 460 – Dynamic Asset Pricing Theory

Usage
-----
    python main.py                      # full pipeline with synthetic data
    python main.py --session evening    # use evening BCT fixing as anchor
    python main.py --no-backtest        # skip backtest (faster)
    python main.py --live               # enter live simulation mode

Steps
-----
    1. Load & align data
    2. Fit basket model (Component 1) + diagnostics
    3. Fit Kalman filter (Component 2) + OU + ECM
    4. Combine into full dataset intrinsic value series
    5. Walk-forward backtest
    6. Model comparison
    7. Live simulation demo
    8. Export results to Excel
    9. Generate all charts
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.loader          import build_master_dataset
from models.basket        import BasketModel
from models.liquidity     import (KalmanSpreadFilter,
                                  OrnsteinUhlenbeckModel,
                                  ErrorCorrectionModel)
from models.realtime      import RealTimeEngine
from backtest.engine      import WalkForwardBacktest, compare_models
from utils.visualize      import save_all_charts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TND Intrinsic Value Model")
    p.add_argument("--start",       default="2020-01-01")
    p.add_argument("--end",         default="2024-12-31")
    p.add_argument("--session",     default="morning",
                   choices=["morning", "evening"])
    p.add_argument("--no-backtest", action="store_true")
    p.add_argument("--live",        action="store_true")
    p.add_argument("--output-dir",  default="output")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 – Data
# ---------------------------------------------------------------------------

def step_data(args) -> pd.DataFrame:
    print("\n" + "━" * 60)
    print("  STEP 1 │ DATA LOADING & ALIGNMENT")
    print("━" * 60)
    data = build_master_dataset(
        start           = args.start,
        end             = args.end,
        fixing_session  = args.session,
    )
    print(f"\n  Columns : {list(data.columns)}")
    print(f"  Shape   : {data.shape}")
    print(f"  NaN     : {data.isna().sum().sum()} total\n")
    return data


# ---------------------------------------------------------------------------
# Step 2 – Basket model
# ---------------------------------------------------------------------------

def step_basket(data: pd.DataFrame) -> BasketModel:
    print("━" * 60)
    print("  STEP 2 │ BASKET WEIGHT ESTIMATION  (Component 1)")
    print("━" * 60)

    basket = BasketModel(
        constrain_weights = False,
        rolling_window    = 90,
    )
    basket.fit(data)

    d = basket.diagnostics()
    if d["R_squared"] < 0.5:
        print("  ⚠  R² < 0.5: the basket explains limited variation.\n"
              "     This may indicate structural breaks or data issues.")
    return basket


# ---------------------------------------------------------------------------
# Step 3 – Liquidity adjustment models
# ---------------------------------------------------------------------------

def step_liquidity(data: pd.DataFrame):
    print("\n" + "━" * 60)
    print("  STEP 3 │ LIQUIDITY ADJUSTMENT MODELS  (Component 2)")
    print("━" * 60)

    # Ornstein-Uhlenbeck
    print("\n  [3a] Ornstein-Uhlenbeck Model")
    ou = OrnsteinUhlenbeckModel(session_split=True)
    ou.fit(data)

    # Kalman Filter  ← primary model
    print("\n  [3b] Kalman Spread Filter  (primary)")
    kalman = KalmanSpreadFilter(estimate_noise=True)
    kalman.fit(data)

    # Error-Correction Model
    print("\n  [3c] Error-Correction Model (VECM)")
    try:
        ecm = ErrorCorrectionModel(lags=2)
        ecm.fit(data)
    except Exception as e:
        print(f"  ECM failed (continuing without it): {e}")
        ecm = None

    return ou, kalman, ecm


# ---------------------------------------------------------------------------
# Step 4 – Combine into full intrinsic value series
# ---------------------------------------------------------------------------

def step_combine(data: pd.DataFrame,
                 basket: BasketModel,
                 kalman: KalmanSpreadFilter,
                 ou:     OrnsteinUhlenbeckModel) -> pd.DataFrame:
    print("\n" + "━" * 60)
    print("  STEP 4 │ GENERATING INTRINSIC VALUE SERIES")
    print("━" * 60)

    # Apply basket (adds predicted_fixing)
    enriched = basket.apply_to_dataset(data)

    # Apply Kalman (adds intrinsic_kf)
    enriched = kalman.apply_to_dataset(enriched)

    # Apply OU
    enriched = ou.apply_to_dataset(enriched)

    # Summary stats
    ib   = enriched["ib_rate"]
    iv   = enriched["intrinsic_kf"].dropna()
    corr = ib.corr(iv)
    print(f"\n  Intrinsic rate range : {iv.min():.4f} – {iv.max():.4f}")
    print(f"  Mean intrinsic       : {iv.mean():.4f}")
    print(f"  Correlation with IB  : {corr:.4f}")
    print(f"  Rows with intrinsic  : {len(iv)}/{len(enriched)}")

    return enriched


# ---------------------------------------------------------------------------
# Step 5 – Walk-forward backtest
# ---------------------------------------------------------------------------

def step_backtest(enriched: pd.DataFrame):
    print("\n" + "━" * 60)
    print("  STEP 5 │ WALK-FORWARD BACKTEST")
    print("━" * 60)

    bt = WalkForwardBacktest(
        train_size  = 252,
        step_size   = 21,
        horizon     = 1,
        mode        = "expanding",
    )
    result = bt.run(enriched)
    return result


# ---------------------------------------------------------------------------
# Step 6 – Model comparison
# ---------------------------------------------------------------------------

def step_compare(enriched: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "━" * 60)
    print("  STEP 6 │ MODEL COMPARISON")
    print("━" * 60)
    return compare_models(enriched, train_frac=0.75)


# ---------------------------------------------------------------------------
# Step 7 – Live simulation
# ---------------------------------------------------------------------------

def step_live(enriched: pd.DataFrame,
              basket: BasketModel,
              kalman: KalmanSpreadFilter,
              n_ticks: int = 20):
    print("\n" + "━" * 60)
    print("  STEP 7 │ LIVE SIMULATION DEMO")
    print("━" * 60)

    # Bootstrap from last 252 days of data
    recent    = enriched.tail(252)
    last_row  = recent.iloc[-1]

    engine = RealTimeEngine(
        basket_model  = basket,
        kalman_model  = kalman,
        fixing_anchor = float(last_row["fixing"]),
        anchor_fx     = {
            "EURUSD": float(last_row["EURUSD"]),
            "GBPUSD": float(last_row["GBPUSD"]),
            "USDJPY": float(last_row["USDJPY"]),
        },
        last_ib_rate  = float(last_row["ib_rate"]),
        init_state    = float(last_row.get("delta_estimate_kf",
                                            last_row["spread_raw"])),
        init_variance = float(kalman.Q_),
    )

    print(f"\n  Replaying last {n_ticks} ticks …\n")
    print(f"  {'Date':<12} {'Fixing':>10} {'Baseline':>10} "
          f"{'δ̂(t)':>10} {'V*(t)':>10} {'IB Rate':>10}")
    print("  " + "-" * 65)

    rng = np.random.default_rng(99)

    for i, (ts, row) in enumerate(recent.tail(n_ticks).iterrows()):
        # Simulate slight intraday FX noise around daily close
        fx_noise = rng.normal(0, 0.0002, 3)
        current_fx = {
            "EURUSD": float(row["EURUSD"]) * (1 + fx_noise[0]),
            "GBPUSD": float(row["GBPUSD"]) * (1 + fx_noise[1]),
            "USDJPY": float(row["USDJPY"]) * (1 + fx_noise[2]),
        }

        # Randomly inject an intraday signal 30% of the time
        signal = (float(row["spread_raw"]) + rng.normal(0, 0.0003)
                  if rng.random() < 0.30 else None)

        state = engine.update(
            current_fx      = current_fx,
            timestamp       = pd.Timestamp(ts).to_pydatetime(),
            intraday_signal = signal,
            signal_noise    = kalman.R_ * 2 if signal else None,
        )

        sig_marker = "●" if signal else " "
        print(f"  {str(ts.date()):<12} "
              f"{state.fixing_anchor:>10.4f} "
              f"{state.baseline:>10.4f} "
              f"{state.delta_mean:>10.5f} "
              f"{state.intrinsic:>10.4f} "
              f"{row['ib_rate']:>10.4f}  {sig_marker}")

        time.sleep(0.03)   # pacing

    hist = engine.history_dataframe()
    print(f"\n  Engine history: {len(hist)} states accumulated.")
    return engine


# ---------------------------------------------------------------------------
# Step 8 – Export to Excel
# ---------------------------------------------------------------------------

def step_export(enriched: pd.DataFrame,
                backtest_result,
                comparison_df: pd.DataFrame,
                basket: BasketModel,
                output_dir: str):
    print("\n" + "━" * 60)
    print("  STEP 8 │ EXCEL EXPORT")
    print("━" * 60)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "tnd_intrinsic_model_results.xlsx")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Sheet 1: Full dataset
        enriched.to_excel(writer, sheet_name="Full Dataset")

        # Sheet 2: Model diagnostics
        diag = basket.diagnostics()
        diag_df = pd.DataFrame([{
            "Metric": k,
            "Value" : v if not isinstance(v, dict) else str(v)
        } for k, v in diag.items()])
        diag_df.to_excel(writer, sheet_name="Basket Diagnostics", index=False)

        # Sheet 3: Rolling weights
        if (basket.result_ and basket.result_.rolling_weights is not None
                and not basket.result_.rolling_weights.empty):
            basket.result_.rolling_weights.to_excel(
                writer, sheet_name="Rolling Weights")

        # Sheet 4: Backtest predictions
        if backtest_result is not None:
            backtest_result.predictions.to_excel(
                writer, sheet_name="Backtest Predictions")

            # Sheet 5: OOS metrics
            m = backtest_result.metrics_oos
            pd.DataFrame([m]).T.rename(columns={0: "Value"}).to_excel(
                writer, sheet_name="OOS Metrics")

        # Sheet 6: Model comparison
        if comparison_df is not None:
            comparison_df.to_excel(writer, sheet_name="Model Comparison")

    print(f"  Results exported → {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "═" * 60)
    print("  TND INTRINSIC VALUE MODEL  |  FIN 460")
    print("  Dynamic Asset Pricing Theory")
    print("═" * 60)

    # --- Pipeline ---
    data     = step_data(args)
    basket   = step_basket(data)
    ou, kalman, ecm = step_liquidity(data)
    enriched = step_combine(data, basket, kalman, ou)

    backtest_result = None
    comparison_df   = None

    if not args.no_backtest:
        backtest_result = step_backtest(enriched)
        comparison_df   = step_compare(enriched)

    if args.live:
        engine = step_live(enriched, basket, kalman)

    step_export(enriched, backtest_result, comparison_df, basket, args.output_dir)

    # --- Charts ---
    print("\n" + "━" * 60)
    print("  STEP 9 │ GENERATING CHARTS")
    print("━" * 60)
    save_all_charts(
        data            = enriched,
        basket_model    = basket,
        kalman_model    = kalman,
        ou_model        = ou,
        backtest_result = backtest_result,
        comparison_df   = comparison_df,
        output_dir      = args.output_dir,
    )

    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  All outputs in: ./{args.output_dir}/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
