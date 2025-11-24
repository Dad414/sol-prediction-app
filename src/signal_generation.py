"""Generate trading signals and explanations from trained model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from feature_engineering import compute_features


@dataclass
class SignalResult:
    timestamp: pd.Timestamp
    direction: str
    probability: float
    probabilities: Dict[str, float]
    close_price: float
    atr: float
    atr_pct: float
    suggested_stop: float
    suggested_target: float
    explanation: str
    supporting_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "probability": self.probability,
            "probabilities": self.probabilities,
            "close_price": self.close_price,
            "atr": self.atr,
            "atr_pct": self.atr_pct,
            "suggested_stop": self.suggested_stop,
            "suggested_target": self.suggested_target,
            "explanation": self.explanation,
            "supporting_metrics": self.supporting_metrics,
        }


import xgboost as xgb
import pickle

def _load_model(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_path.suffix == ".json":
        # Load XGBoost model
        metadata_path = model_path.with_suffix(".metadata.pkl")
        if not metadata_path.exists():
             raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        return {
            "model": model,
            "features": metadata["features"],
            "timestamp": metadata["timestamp"],
            "classes": metadata["classes"]
        }
        
    data = pd.read_pickle(model_path)
    if not isinstance(data, dict) or "model" not in data or "features" not in data:
        raise ValueError("Model artifact is malformed.")
    
    if "classes" not in data and hasattr(data["model"], "classes_"):
        data["classes"] = data["model"].classes_
        
    return data


def _load_feature_frame(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["open_time", "close_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    feature_df = compute_features(df)
    return feature_df


def _format_explanation(
    direction: str,
    probability: float,
    row: pd.Series,
    supporting_metrics: Dict[str, Any],
) -> str:
    clauses = []
    if direction == "long":
        clauses.append("Momentum tilts bullish with upside expectation over the next 30 minutes.")
    elif direction == "short":
        clauses.append("Momentum leans bearish with downside pressure anticipated in the next 30 minutes.")
    else:
        clauses.append("Signals are mixed; stand aside unless new information emerges.")

    if "rsi_14" in row:
        if row["rsi_14"] >= 65:
            clauses.append(f"RSI sits at {row['rsi_14']:.1f}, signaling overbought strength.")
        elif row["rsi_14"] <= 35:
            clauses.append(f"RSI prints {row['rsi_14']:.1f}, indicating oversold momentum.")
        else:
            clauses.append(f"RSI is neutral at {row['rsi_14']:.1f}.")

    if "macd_diff" in row:
        if row["macd_diff"] > 0:
            clauses.append("MACD histogram is positive, suggesting bullish momentum.")
        else:
            clauses.append("MACD histogram is negative, suggesting bearish momentum.")

    if "stoch_k" in row and "stoch_d" in row:
        if row["stoch_k"] > 80:
            clauses.append(f"Stochastic K is {row['stoch_k']:.1f} (overbought).")
        elif row["stoch_k"] < 20:
            clauses.append(f"Stochastic K is {row['stoch_k']:.1f} (oversold).")

    ema_keys = [key for key in row.index if key.startswith("ema_distance_")]
    if ema_keys:
        dominant = max(ema_keys, key=lambda k: abs(row[k]))
        span = dominant.split("_")[-1]
        pct = row[dominant] * 100
        if pct > 0:
            clauses.append(f"Price trades {pct:.2f}% above the EMA{span}, showing trend strength.")
        else:
            clauses.append(f"Price trades {pct:.2f}% below the EMA{span}, reflecting trend weakness.")

    if "bb_position" in row:
        if row["bb_position"] >= 0.7:
            clauses.append("Price hugs the upper Bollinger band, confirming bullish pressure.")
        elif row["bb_position"] <= 0.3:
            clauses.append("Price is near the lower Bollinger band, underscoring bearish tone.")

    clauses.append(f"Model confidence: {probability:.2%}.")
    if "atr_pct" in supporting_metrics:
        clauses.append(f"ATR implies typical 30-min range of {supporting_metrics['atr_pct']:.2%} relative to price.")

    return " ".join(clauses)


def generate_signal(
    timestamp: Optional[str] = None,
    *,
    data_path: Path,
    model_path: Path,
) -> SignalResult:
    artifact = _load_model(model_path)
    model = artifact["model"]
    feature_names = artifact["features"]

    feature_df = _load_feature_frame(data_path)

    if timestamp:
        ts = pd.Timestamp(timestamp, tz="UTC")
        row = feature_df.loc[feature_df["open_time"] == ts]
        if row.empty:
            raise ValueError(f"No candle found for timestamp {ts}. Ensure the dataset contains this interval.")
        row = row.iloc[0]
    else:
        row = feature_df.iloc[-1]
        ts = row["open_time"]

    X = row[feature_names].values.reshape(1, -1)
    probabilities = model.predict_proba(X)[0]
    classes = artifact["classes"]
    prob_map = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    direction = classes[int(np.argmax(probabilities))]
    probability = prob_map[direction]

    atr = float(row.get("atr_14", np.nan))
    atr_pct = float(row.get("atr_pct", np.nan))
    close_price = float(row["close"])

    risk_multiple_stop = 1.2
    risk_multiple_target = 2.0

    if direction == "long":
        suggested_stop = close_price - risk_multiple_stop * atr
        suggested_target = close_price + risk_multiple_target * atr
    elif direction == "short":
        suggested_stop = close_price + risk_multiple_stop * atr
        suggested_target = close_price - risk_multiple_target * atr
    else:
        suggested_stop = close_price
        suggested_target = close_price

    supporting_metrics = {
        "close": close_price,
        "atr": atr,
        "atr_pct": atr_pct,
        "rsi_14": float(row.get("rsi_14", np.nan)),
        "macd": float(row.get("macd", np.nan)),
        "macd_signal": float(row.get("macd_signal", np.nan)),
        "macd_diff": float(row.get("macd_diff", np.nan)),
        "stoch_k": float(row.get("stoch_k", np.nan)),
        "stoch_d": float(row.get("stoch_d", np.nan)),
        "ema_distance_10": float(row.get("ema_distance_10", np.nan)),
        "bb_position": float(row.get("bb_position", np.nan)),
        "probabilities": prob_map,
    }

    explanation = _format_explanation(direction, probability, row, supporting_metrics)

    return SignalResult(
        timestamp=ts,
        direction=direction,
        probability=probability,
        probabilities=prob_map,
        close_price=close_price,
        atr=atr,
        atr_pct=atr_pct,
        suggested_stop=suggested_stop,
        suggested_target=suggested_target,
        explanation=explanation,
        supporting_metrics=supporting_metrics,
    )


def print_signal(signal: SignalResult, output_json: bool = False) -> None:
    if output_json:
        print(json.dumps(signal.to_dict(), indent=2))
        return

    print(f"Timestamp (UTC): {signal.timestamp}")
    print(f"Suggested stance: {signal.direction.upper()} (confidence {signal.probability:.2%})")
    print(f"Last close: {signal.close_price:.4f} | ATR: {signal.atr:.4f} ({signal.atr_pct:.2%})")
    print(f"Risk box: stop {signal.suggested_stop:.4f}, target {signal.suggested_target:.4f}")
    print("Probabilities:")
    for label, prob in signal.probabilities.items():
        print(f"  - {label}: {prob:.2%}")
    print("Explanation:")
    print(f"  {signal.explanation}")


__all__ = ["SignalResult", "generate_signal", "print_signal"]


