"""Streamlit dashboard for SOL/USDT 30-minute trend predictions."""

from __future__ import annotations

import json
import sys
import time
import pickle
from pathlib import Path
from typing import Optional

import xgboost as xgb

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_option_menu import option_menu

# Add project root to path to allow importing from data
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.binance_fetch import update_history
from feature_engineering import compute_features
from signal_generation import generate_signal
from src.auth import check_authentication, logout, get_users

DATA_PATH = Path("data/raw/solusdt_30m.csv")
MODEL_PATH = Path("models/sol_trend_xgb.json")
METRICS_PATH = Path("reports/model_metrics_xgb.json")


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
            "Trained model artifact not found. Please run `python src/train_xgb.py` first."
        )
    
    if path.suffix == ".json":
        # Load XGBoost model
        metadata_path = path.with_suffix(".metadata.pkl")
        if not metadata_path.exists():
             raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            
        model = xgb.XGBClassifier()
        model.load_model(path)
        
        return {
            "model": model,
            "features": metadata["features"],
            "timestamp": metadata["timestamp"],
            "classes": metadata["classes"]
        }

    data = pd.read_pickle(path)
    # Ensure classes are available for RF models too
    if "classes" not in data and hasattr(data["model"], "classes_"):
        data["classes"] = data["model"].classes_
    return data


def inject_custom_css():
    st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #1E1E1E;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #9945FF; /* Solana Purple */
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0E0E10;
        }

        /* Headers */
        h1, h2, h3 {
            color: #FFFFFF;
        }
        h1 {
            background: -webkit-linear-gradient(45deg, #9945FF, #14F195);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #1E1E1E;
            border-radius: 5px;
            color: #FFF;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #9945FF;
            color: #FFF;
        }
    </style>
    """, unsafe_allow_html=True)


def build_history_table(feature_store: pd.DataFrame, artifact) -> pd.DataFrame:
    model = artifact["model"]
    feature_cols = artifact["features"]

    if feature_store.empty:
        return pd.DataFrame()

    # Ensure all feature columns exist
    missing_cols = [c for c in feature_cols if c not in feature_store.columns]
    if missing_cols:
        st.warning(f"Missing features in data: {missing_cols}. Re-running feature engineering...")
        return pd.DataFrame()

    X = feature_store[feature_cols]
    probs = model.predict_proba(X)
    preds = model.predict(X)

    prob_df = pd.DataFrame(probs, columns=[f"prob_{cls}" for cls in artifact["classes"]])
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
    window_size = 100
    idx = df[df["open_time"] == selected_ts].index[0]
    start_idx = max(0, idx - window_size)
    end_idx = min(len(df), idx + 20)
    
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
            go.Scatter(x=chart_data["open_time"], y=chart_data["ema_10"], line=dict(color='#FF9F1C', width=1), name="EMA 10"),
            row=1, col=1
        )
    if "ema_40" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["ema_40"], line=dict(color='#2EC4B6', width=1), name="EMA 40"),
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
            go.Bar(x=chart_data["open_time"], y=chart_data["macd_diff"], name="MACD Hist", marker_color=np.where(chart_data["macd_diff"] < 0, '#FF5252', '#69F0AE')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["macd"], line=dict(color='#2EC4B6', width=1), name="MACD"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["macd_signal"], line=dict(color='#FF9F1C', width=1), name="Signal"),
            row=2, col=1
        )

    # RSI
    if "rsi_14" in chart_data.columns:
        fig.add_trace(
            go.Scatter(x=chart_data["open_time"], y=chart_data["rsi_14"], line=dict(color='#9945FF', width=1), name="RSI"),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#69F0AE", row=3, col=1)

    # Highlight selected candle
    fig.add_vline(x=selected_ts, line_width=2, line_dash="dash", line_color="yellow")

    fig.update_layout(
        title="SOL/USDT Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
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

    st.markdown("### 🤖 Model Signal")
    
    # Confidence Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = signal.probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence ({signal.direction.upper()})"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#9945FF"},
            'steps': [
                {'range': [0, 50], 'color': "#333"},
                {'range': [50, 80], 'color': "#555"},
                {'range': [80, 100], 'color': "#777"}],
            'threshold': {
                'line': {'color': "#14F195", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"}
    )
    
    col_gauge, col_metrics = st.columns([1, 2])
    
    with col_gauge:
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col_metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Last Close", f"${signal.close_price:.2f}")
        col2.metric("ATR (30m)", f"{signal.atr:.4f}", delta=f"{signal.atr_pct:.2%}")
        target_delta = signal.suggested_target - signal.close_price
        stop_delta = signal.close_price - signal.suggested_stop
        col3.metric("R:R Ratio", f"{target_delta:.2f} / {stop_delta:.2f}")
        
        st.markdown("**💡 Trend Explanation**")
        st.info(signal.explanation)

    st.markdown("### 📊 Probability Distribution")
    prob_df = pd.DataFrame(
        {"label": list(signal.probabilities.keys()), "probability": list(signal.probabilities.values())}
    )
    st.bar_chart(prob_df, x="label", y="probability", height=200)

    st.markdown("### 🎯 Trade Setup")
    
    col_entry, col_stop, col_target = st.columns(3)
    with col_entry:
        st.markdown(f"<div style='text-align: center; padding: 10px; background: #333; border-radius: 5px;'><h4>Entry</h4><h2>${signal.close_price:.2f}</h2></div>", unsafe_allow_html=True)
    with col_stop:
        st.markdown(f"<div style='text-align: center; padding: 10px; background: #333; border-radius: 5px; border: 1px solid #FF5252;'><h4>Stop Loss</h4><h2 style='color: #FF5252'>${signal.suggested_stop:.2f}</h2></div>", unsafe_allow_html=True)
    with col_target:
        st.markdown(f"<div style='text-align: center; padding: 10px; background: #333; border-radius: 5px; border: 1px solid #14F195;'><h4>Target</h4><h2 style='color: #14F195'>${signal.suggested_target:.2f}</h2></div>", unsafe_allow_html=True)

    st.markdown("### 📉 Key Metrics")
    st.table(pd.DataFrame(signal.supporting_metrics, index=["value"]).T)


def main() -> None:
    st.set_page_config(
        page_title="SOL/USDT 30m Trend Model",
        layout="wide",
        page_icon="🔮",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    
    # Authentication Check
    user_email = check_authentication()
    if not user_email:
        st.stop()

    # Sidebar Navigation
    with st.sidebar:
        st.image("https://cryptologos.cc/logos/solana-sol-logo.png", width=50)
        st.title("SOL Predictor")
        st.markdown(f"Logged in as: **{user_email}**")
        
        selected = option_menu(
            menu_title="Menu",
            options=["Dashboard", "Analysis", "History", "Settings"],
            icons=["speedometer2", "graph-up-arrow", "clock-history", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#0E0E10"},
                "icon": {"color": "#14F195", "font-size": "20px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#333"},
                "nav-link-selected": {"background-color": "#9945FF"},
            }
        )
        
        if st.button("Logout", type="secondary"):
            logout()

    try:
        artifact = load_artifact(MODEL_PATH)
        feature_store = load_feature_store(DATA_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if selected == "Dashboard":
        st.title("🚀 Live Dashboard")
        st.caption("Real-time market insights and model predictions.")
        
        if not feature_store.empty:
            timestamps = feature_store["open_time"]
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize("UTC")
            else:
                timestamps = timestamps.dt.tz_convert("UTC")
            options = [ts.isoformat() for ts in timestamps]
            latest_option = options[-1] if options else None
            
            # Show latest signal by default
            render_signal_details(latest_option, feature_store)
        else:
            st.warning("No data available. Please fetch data in Settings.")

    elif selected == "Analysis":
        st.title("📈 Deep Dive Analysis")
        st.caption("Inspect historical candles and model performance.")
        
        if not feature_store.empty:
            timestamps = feature_store["open_time"]
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize("UTC")
            else:
                timestamps = timestamps.dt.tz_convert("UTC")
            options = [ts.isoformat() for ts in timestamps]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_ts = st.selectbox(
                    "Select Candle (UTC)",
                    options=options,
                    index=len(options) - 1 if options else 0,
                )
            
            render_signal_details(selected_ts, feature_store)

    elif selected == "History":
        st.title("📜 Signal History")
        if not feature_store.empty:
            history = build_history_table(feature_store, artifact)

            if not history.empty:
                st.dataframe(
                    history.sort_values("open_time", ascending=False).head(100),
                    use_container_width=True,
                    height=600
                )

                csv_data = history.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="solusdt_signal_history.csv",
                    mime="text/csv",
                )
            else:
                st.warning("Could not generate history table.")
        else:
            st.info("No historical data to display.")

    elif selected == "Settings":
        st.title("⚙️ Settings")
        
        st.markdown("### 🔄 Data Management")
        if st.button("Refresh Data from Binance"):
            with st.spinner("Fetching latest data..."):
                try:
                    update_history(output_path=DATA_PATH)
                    st.success("Data updated successfully!")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
        
        st.divider()
        st.markdown("### 📊 Model Metrics")
        metrics_data = load_metrics(METRICS_PATH)
        if metrics_data:
            st.json(metrics_data)
        else:
            st.info("Metrics file not found.")
            
        st.divider()
        st.markdown("### 👥 User Management")
        
        # Admin-only view
        ADMIN_EMAIL = "dadisworking414@gmail.com"
        
        if user_email == ADMIN_EMAIL:
            with st.expander("View Registered Users (Admin Only)"):
                users = get_users()
                if users:
                    st.table(pd.DataFrame(users, columns=["Registered Emails"]))
                else:
                    st.info("No users found.")
        else:
            st.info("User management is restricted to administrators.")


if __name__ == "__main__":
    main()
