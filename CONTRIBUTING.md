# Contributing

Contributions, corrections, and extensions are welcome.

## Getting started

```bash
git clone https://github.com/YOUR_USERNAME/tnd-intrinsic-value-model.git
cd tnd-intrinsic-value-model
pip install -r requirements.txt
```

## Suggested contributions

- **Real BCT data pipeline** — automate scraping of BCT fixing and IB rates
- **Intraday FX feed** — integrate a tick-data provider (e.g. OANDA, Interactive Brokers)
- **Regime-switching spread model** — HMM with two liquidity regimes
- **ML spread forecaster** — replace Kalman with a gradient-boosted model on lagged features
- **Dashboard** — Streamlit or Dash app for live monitoring

## Code style

- Follow PEP 8. Use 4-space indentation.
- Add docstrings to all public functions.
- Keep module-level state minimal — prefer classes.
- New models should implement `.fit(data)` and `.apply(data)` for compatibility with the pipeline.

## Pull request checklist

- [ ] `run_standalone.py` runs end-to-end without errors
- [ ] New functions have docstrings
- [ ] Added entry to `CHANGELOG.md` if applicable
