# Changelog

## [1.0.0] — 2025

### Added
- Basket weight estimation via OLS with HAC-robust errors and rolling 90-day window
- Ornstein-Uhlenbeck spread model with exact MLE calibration
- Kalman filter spread nowcaster with EM parameter estimation (RTS smoother)
- Error-Correction Model (VECM) for cointegrated IB/fixing dynamics
- Real-time engine combining both components with intraday update support
- Walk-forward backtest with expanding window (252-day initial, 21-day step)
- Model comparison across all three spread models
- 6 diagnostic charts + Excel export
- Self-contained standalone runner (numpy/scipy only)
- Synthetic data generator calibrated to realistic TND dynamics
- Interactive Jupyter notebook for exploration
