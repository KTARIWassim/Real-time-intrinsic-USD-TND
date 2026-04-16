"""
data/loader.py
--------------
Fetches and aligns all input data required by the TND intrinsic value model:
  - Global FX pairs: EUR/USD, GBP/USD, USD/JPY  (via yfinance)
  - BCT official fixing rates                    (CSV / manual input)
  - BCT interbank (IB) rates                     (CSV / manual input, t-1 lag)

The BCT does not have a public API, so fixing and IB data are loaded from
CSV files that you populate manually from the BCT website.  A synthetic
dataset is generated automatically if those files do not exist, so the
pipeline runs end-to-end out of the box for development and testing.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.dirname(__file__)
FIXING_CSV = os.path.join(DATA_DIR, "bct_fixing.csv")
IB_CSV     = os.path.join(DATA_DIR, "bct_interbank.csv")


# ---------------------------------------------------------------------------
# 1.  Global FX data (public, via yfinance)
# ---------------------------------------------------------------------------

FX_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}


def fetch_global_fx(start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for EUR/USD, GBP/USD, USD/JPY from Yahoo Finance.

    Parameters
    ----------
    start, end : str   ISO date strings, e.g. '2020-01-01'
    interval   : str   '1d', '1h', '5m', etc.

    Returns
    -------
    DataFrame with DatetimeIndex and columns [EURUSD, GBPUSD, USDJPY]
    containing closing prices.
    """
    frames = {}
    for name, ticker in FX_TICKERS.items():
        raw = yf.download(ticker, start=start, end=end, interval=interval,
                          progress=False, auto_adjust=True)
        if raw.empty:
            raise RuntimeError(f"No data returned for {ticker}. "
                               "Check your internet connection or ticker symbol.")
        frames[name] = raw["Close"].squeeze()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "datetime"
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# 2.  BCT fixing rates  (manual CSV or synthetic fallback)
# ---------------------------------------------------------------------------

def load_bct_fixing(path: str = FIXING_CSV) -> pd.DataFrame:
    """
    Load the BCT official USD/TND fixing rates.

    Expected CSV format
    -------------------
    date,session,rate
    2020-01-02,morning,3.1020
    2020-01-02,evening,3.1045
    ...

    'session' must be one of: morning, evening.
    Returns a DataFrame indexed by (date, session).
    """
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        df["session"] = df["session"].str.lower().str.strip()
        df = df.set_index(["date", "session"]).sort_index()
        return df[["rate"]]
    else:
        print(f"[loader] {path} not found – generating synthetic BCT fixing data.")
        return _synthetic_fixing()


def _synthetic_fixing(
    start: str = "2020-01-01",
    end:   str = "2024-12-31",
    base:  float = 3.10,
    seed:  int   = 42,
) -> pd.DataFrame:
    """
    Generate plausible synthetic BCT USD/TND fixing data for development.
    Uses a random-walk with a mild drift and two daily sessions.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)          # business days only
    n     = len(dates)

    # Simulate a slow drift (TND gradual depreciation) plus noise
    log_returns = rng.normal(0.00008, 0.0012, n)
    log_prices  = np.cumsum(log_returns) + np.log(base)
    morning_rates = np.exp(log_prices)

    # Evening fixing is very close to morning but with a small intraday move
    evening_rates = morning_rates * np.exp(rng.normal(0.00005, 0.0003, n))

    records = []
    for i, d in enumerate(dates):
        records.append({"date": d, "session": "morning", "rate": morning_rates[i]})
        records.append({"date": d, "session": "evening", "rate": evening_rates[i]})

    df = pd.DataFrame(records).set_index(["date", "session"]).sort_index()
    return df


# ---------------------------------------------------------------------------
# 3.  BCT interbank (IB) rates  (manual CSV or synthetic fallback)
# ---------------------------------------------------------------------------

def load_bct_interbank(path: str = IB_CSV) -> pd.Series:
    """
    Load the BCT published average interbank USD/TND rate.

    Expected CSV format
    -------------------
    date,ib_rate
    2020-01-02,3.1035
    ...

    IMPORTANT: the BCT publishes IB(t) on day t+1, so the value labelled
    for date t reflects trading on date t-1 in practice.  This function
    loads the values as-labelled; the alignment/lag handling is done in
    the spread analysis module.

    Returns a Series indexed by date.
    """
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return df["ib_rate"].sort_index()
    else:
        print(f"[loader] {path} not found – generating synthetic IB rate data.")
        return _synthetic_ib()


def _synthetic_ib(seed: int = 42) -> pd.Series:
    """
    Synthetic IB rates: fixing morning + small stochastic spread.
    Spread is mean-reverting around 0.002 TND, with occasional spikes.
    """
    fixing_df = _synthetic_fixing(seed=seed)
    morning   = fixing_df.xs("morning", level="session")["rate"]

    rng = np.random.default_rng(seed + 1)
    n   = len(morning)

    # OU spread: θ=0.3, μ=0.002, σ=0.0015
    spread = np.empty(n)
    spread[0] = 0.002
    theta, mu, sigma = 0.3, 0.002, 0.0015
    for i in range(1, n):
        spread[i] = (spread[i-1]
                     + theta * (mu - spread[i-1])
                     + sigma * rng.normal())

    # Occasional liquidity spikes
    spike_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    spread[spike_idx] += rng.normal(0.005, 0.003, len(spike_idx))

    ib = morning.values + np.abs(spread)   # IB always ≥ fixing (scarcity premium)
    return pd.Series(ib, index=morning.index, name="ib_rate")


# ---------------------------------------------------------------------------
# 4.  Master dataset builder
# ---------------------------------------------------------------------------

def build_master_dataset(
    start: str = "2020-01-01",
    end:   str = "2024-12-31",
    fixing_session: str = "morning",
) -> pd.DataFrame:
    """
    Assemble one aligned daily DataFrame used by all downstream modules.

    Columns
    -------
    EURUSD, GBPUSD, USDJPY : global FX closing prices
    fixing                  : BCT official USD/TND (chosen session)
    ib_rate                 : BCT interbank USD/TND (as-published, t-1 lag)
    spread_raw              : ib_rate − fixing  (raw observable spread)
    log_ret_EURUSD, ...     : log-returns of each global FX pair
    log_ret_fixing          : log-return of the fixing rate

    Parameters
    ----------
    fixing_session : 'morning' or 'evening' – which BCT session to anchor on
    """
    print("[loader] Fetching global FX data …")
    fx = fetch_global_fx(start, end, interval="1d")

    print("[loader] Loading BCT fixing rates …")
    fixing_df = load_bct_fixing()
    if fixing_session not in ("morning", "evening"):
        raise ValueError("fixing_session must be 'morning' or 'evening'")
    try:
        fixing = fixing_df.xs(fixing_session, level="session")["rate"]
    except KeyError:
        fixing = fixing_df.xs("morning", level="session")["rate"]
    fixing.index = pd.to_datetime(fixing.index)

    print("[loader] Loading BCT interbank rates …")
    ib = load_bct_interbank()
    ib.index = pd.to_datetime(ib.index)

    # Align all series on business-day index
    master = fx.copy()
    master = master.join(fixing.rename("fixing"), how="left")
    master = master.join(ib.rename("ib_rate"), how="left")
    master.ffill(inplace=True)
    master.dropna(inplace=True)

    # Log-returns
    for col in ["EURUSD", "GBPUSD", "USDJPY", "fixing"]:
        master[f"log_ret_{col}"] = np.log(master[col]).diff()

    # Observable spread (note: this is S(t-1) from the model's perspective)
    master["spread_raw"] = master["ib_rate"] - master["fixing"]

    master.dropna(inplace=True)

    print(f"[loader] Master dataset: {len(master)} rows, "
          f"{master.index[0].date()} → {master.index[-1].date()}")
    return master
