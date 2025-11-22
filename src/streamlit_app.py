"""Streamlit dashboard for SOL/USDT 30-minute trend predictions."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to path to allow importing from data
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.binance_fetch import update_history
from feature_engineering import compute_features
from signal_generation import generate_signal

DATA_PATH = Path("data/raw/solusdt_30m.csv")
MODEL_PATH = Path("models/sol_trend_random_forest.pkl")
METRICS_PATH = Path("reports/model_metrics.json")


@st.cache_data(show_spinner=False)
def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    return df.sort_values("open_time").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_feature_store(path: Path) -> pd.DataFrame:
    raw = load_raw_data(path)
    if raw.empty:
        return pd.DataFrame()
    features = compute_features(raw)
    return features


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            "Trained model artifact not found. Please run `python src/train.py` first."
        )
    return pd.read_pickle(path)


def build_history_table(feature_store: pd.DataFrame, artifact) -> pd.DataFrame:
    model = artifact["model"]
    feature_cols = artifact["features"]

    if feature_store.empty:
        return pd.DataFrame()

    # Ensure all feature columns exist
    missing_cols = [c for c in feature_cols if c not in feature_store.columns]
    if missing_cols:
        st.warning(f"Missing features in data: {missing_cols}. Re-running feature engineering...")
        # In a real scenario, we might need to re-compute or fail gracefully
        return pd.DataFrame()

    X = feature_store[feature_cols]
    probs = model.predict_proba(X)
    preds = model.predict(X)

    prob_df = pd.DataFrame(probs, columns=[f"prob_{cls}" for cls in model.classes_])
    history = pd.concat(
        [
            feature_store[["open_time", "close", "atr_14", "atr_pct", "rsi_14"]],
            prob_df,
        ],
        axis=1,
    )
    history["prediction"] = preds
    history["confidence"] = prob_df.max(axis=1)
    return history


def render_chart(df: pd.DataFrame, selected_ts: pd.Timestamp) -> None:
    """Render interactive Plotly chart with indicators."""
    # Filter data to show context around selected timestamp
    window_size = 100
    idx = df[df["open_time"] == selected_ts].index[0]
    start_idx = max(0, idx - window_size)
    end_idx = min(len(df), idx + 20) # Show a bit of future if available (or empty space)
    
    chart_data = df.iloc[start_idx:end_idx]

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=chart_data["open_time"],
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="OHLC"
        ),
        row=1, col=1
    )

    # EMAs
    if "ema_10" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["ema_10"], line=dict(color='orange', width=1), name="EMA 10"),
            row=1, col=1
        )
    if "ema_40" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["ema_40"], line=dict(color='blue', width=1), name="EMA 40"),
            row=1, col=1
        )

    # Bollinger Bands
    if "bb_upper" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["bb_upper"], line=dict(color='gray', width=1, dash='dash'), name="BB Upper"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["bb_lower"], line=dict(color='gray', width=1, dash='dash'), name="BB Lower", fill='tonexty'),
            row=1, col=1
        )

    # MACD
    if "macd" in chart_data.columns:
        fig.add_trace(
            go.Bar(x=chart_data["open_time"], y=chart_data["macd_diff"], name="MACD Hist", marker_color=np.where(chart_data["macd_diff"] < 0, 'red', 'green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["macd"], line=dict(color='blue', width=1), name="MACD"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["macd_signal"], line=dict(color='orange', width=1), name="Signal"),
            row=2, col=1
        )

    # RSI
    if "rsi_14" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["rsi_14"], line=dict(color='purple', width=1), name="RSI"),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Highlight selected candle
    fig.add_vline(x=selected_ts, line_width=2, line_dash="dash", line_color="yellow")

    fig.update_layout(
        title="SOL/USDT Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_signal_details(timestamp: Optional[str], feature_store: pd.DataFrame) -> None:
    try:
        signal = generate_signal(
            timestamp=timestamp,
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
        )
    except (ValueError, FileNotFoundError) as e:
        st.warning(f"Could not generate signal: {e}")
        return

    # Render Chart
    if timestamp:
        ts = pd.Timestamp(timestamp).tz_convert("UTC")
        render_chart(feature_store, ts)

    st.subheader("Trading Signal")
    
    # Confidence Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = signal.probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence ({signal.direction.upper()})"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "gray"},
                {'range': [50, 80], 'color': "lightgray"},
                {'range': [80, 100], 'color': "white"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    col_gauge, col_metrics = st.columns([1, 2])
    
    with col_gauge:
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col_metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Last Close", f"{signal.close_price:.2f}")
        col2.metric("ATR (30m)", f"{signal.atr:.4f}", delta=f"{signal.atr_pct:.2%}")
        target_delta = signal.suggested_target - signal.close_price
        stop_delta = signal.close_price - signal.suggested_stop
        col3.metric("Reward / Risk", f"{target_delta:.2f} / {stop_delta:.2f}")
        
        st.write("**Trend explanation**")
        st.info(signal.explanation)

    st.write("**Probability breakdown**")
    prob_df = pd.DataFrame(
        {"label": list(signal.probabilities.keys()), "probability": list(signal.probabilities.values())}
    )
    st.bar_chart(prob_df, x="label", y="probability", height=200)

    st.write("**Risk box (price levels)**")
    st.table(
        pd.DataFrame(
            {
                "level": ["Entry", "Stop", "Target"],
                "price": [
                    signal.close_price,
                    signal.suggested_stop,
                    signal.suggested_target,
                ],
            }
        )
    )

    st.write("**Key metrics**")
    st.table(pd.DataFrame(signal.supporting_metrics, index=["value"]).T)


def main() -> None:
    st.set_page_config(
        page_title="SOL/USDT 30m Trend Model",
        layout="wide",
        page_icon="📈",
    )
    st.title("SOL/USDT 30-minute Trend Signal")
    st.caption("Model-driven guidance for the next 30-minute window using Binance data.")

    try:
        artifact = load_artifact(MODEL_PATH)
        feature_store = load_feature_store(DATA_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Dashboard", "History", "Controls"])

    with tab1:
        st.sidebar.header("Signal Query")
        if not feature_store.empty:
            timestamps = feature_store["open_time"]
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize("UTC")
            else:
                timestamps = timestamps.dt.tz_convert("UTC")
            options = [ts.isoformat() for ts in timestamps]
            latest_option = options[-1] if options else None

            selected_ts = st.sidebar.selectbox(
                "Candle timestamp (UTC)",
                options=options,
                index=len(options) - 1 if options else 0,
            )
            st.sidebar.write("Pick the candle open time to inspect the model's next-30-min view.")
            
            render_signal_details(selected_ts or latest_option, feature_store)
        else:
            st.warning("No data available. Please fetch data in the Controls tab.")

    with tab2:
        st.subheader("Historical Signals")
        if not feature_store.empty:
            history = build_history_table(feature_store, artifact)

            if not history.empty:
                st.dataframe(
                    history.sort_values("open_time", ascending=False).head(100),
                    use_container_width=True,
                )

                csv_data = history.to_csv(index=False)
                st.download_button(
                    label="Download full signal history (CSV)",
                    data=csv_data,
                    file_name="solusdt_signal_history.csv",
                    mime="text/csv",
                )
            else:
                st.warning("Could not generate history table (likely missing features).")
        else:
            st.info("No historical data to display.")

    with tab3:
        st.subheader("Data & Model Controls")
        
        st.write("### Data Refresh")
        st.write("Fetch the latest 30-minute klines from Binance.")
        
        if st.button("Refresh Data Now"):
            with st.spinner("Fetching latest data from Binance..."):
                try:
                    update_history(output_path=DATA_PATH)
                    st.success("Data updated successfully!")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        st.divider()
        st.subheader("Model Metrics")
        metrics_data = load_metrics(METRICS_PATH)
        if metrics_data:
            st.json(metrics_data)
        else:
            st.info("Metrics file not found. Train the model to generate evaluation metrics.")


if __name__ == "__main__":
    main()

