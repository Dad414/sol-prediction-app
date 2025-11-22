# SOL/USDT 30-Minute Trend Model

## Data Summary
- **Source**: Binance spot klines via `data/binance_fetch.py`.
- **Granularity**: 30-minute candles (`open_time`, `high`, `low`, `close`, `volume`, trades).
- **Storage**: `data/raw/solusdt_30m.csv` with incremental append support.
- **Coverage**: Determined by fetch parameters; update the dataset periodically for freshest signals.

## Feature Engineering
- Momentum: 1/3/6-candle percentage returns, log returns.
- Trend: EMA(5/10/20/40) distances, Bollinger Bands, RSI(14).
- Volatility: ATR(14), ATR percent of price.
- Volume: Rolling mean (10), relative volume, rate of change.
- Calendar: Day of week, hour, minute.
- Processing coded in `src/feature_engineering.py`; NaN rows are dropped after indicator warm-up.

## Labeling
- Ternary targets generated in `src/labeling.py`.
- **Horizon**: next 30-minute close (`horizon=1`).
- **Thresholds**: Long if return ≥ +0.15%, Short if ≤ –0.15%, else Neutral.
- Future returns retained as `future_return` for evaluation.

## Modeling
- Training script `src/train_model.py` loads data → features → labels, splits chronologically (80/20), and fits a class-weighted `RandomForestClassifier`.
- Artifacts persisted to `models/sol_trend_random_forest.pkl` alongside feature list + label config.
- Metrics (macro F1, balanced accuracy, confusion matrix, feature importance) exported to `reports/model_metrics.json`.
- **Prerequisites**: `pip install -r requirements.txt` (includes pandas, numpy, scikit-learn, streamlit).
- **Usage**: `python src/train_model.py --data data/raw/solusdt_30m.csv`.

## Signal Generation & Explanation
- `src/signal_generation.py` wraps inference, converting predictions to direction (long/neutral/short), confidence, ATR-based stop/target, and narrative explanations.
- CLI entry-point: `python src/predict.py --timestamp 2024-11-12T12:00:00Z`.
- Streamlit dashboard (`app/streamlit_app.py`) visualizes latest signal, probability breakdown, risk box, and provides downloadable history.

## Validation Results
> Run `python src/train_model.py` to regenerate metrics, then review `reports/model_metrics.json`.

Key checkpoints once metrics are available:
- Macro F1 ≥ 0.40 and balanced accuracy ≥ 0.50.
- Inspect per-class recall for `short` vs `long`.
- Review top feature importances for over-reliance on single inputs.

## Risk Management Notes
- Default risk box: stop = 1.2 × ATR, target = 2.0 × ATR in direction of signal.
- ATR percent indicates expected 30-minute range; adjust position sizing accordingly.
- Neutral signals advise no trade—respect to avoid chop.

## Next Steps
- Backtest strategy with transaction costs to validate live viability.
- Consider hyperparameter optimization (e.g., cross-validated gradient boosting).
- Explore ensembling with sequence models (LSTM/Transformer) if higher recall needed.
- Automate daily retraining and data refresh via scheduler.


