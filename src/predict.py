"""CLI entry-point for generating SOL/USDT trend signals."""

from __future__ import annotations
import argparse
from pathlib import Path

from signal_generation import generate_signal, print_signal

DEFAULT_DATA_PATH = Path("data/raw/solusdt_30m.csv")
DEFAULT_MODEL_PATH = Path("models/sol_trend_random_forest.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict SOL/USDT trend for the next 30 minutes.")
    parser.add_argument(
        "--timestamp",
        help="ISO8601 timestamp (UTC) referencing the open time of the candle. Defaults to latest available.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to raw candles CSV.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained model artifact.")
    parser.add_argument("--json", action="store_true", help="Output response as JSON.")
    return parser.parse_args()


# ---------------------------
# INTERNAL: convert the signal dict into clean Telegram-bot output
# ---------------------------
def _make_output(signal: dict) -> dict:
    """Convert CLI-style signal dict to a clean structured result for Telegram bot."""
    return {
        "timestamp": signal.get("timestamp"),
        "stance": signal.get("stance"),
        "confidence": signal.get("confidence"),
        "last_close": signal.get("last_close"),
        "atr": signal.get("atr"),
        "atr_pct": signal.get("atr_pct"),
        "stop": signal.get("stop_loss"),
        "target": signal.get("take_profit"),
        "probs": signal.get("probabilities"),
        "explanation": signal.get("explanation"),
    }


# ---------------------------
# PUBLIC API: used by Telegram bot
# ---------------------------
def run_prediction(
    timestamp: str | None = None,
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Run a prediction and return a clean dict (no printing).

    This is what the Telegram bot imports.
    """
    signal = generate_signal(
        timestamp=timestamp,
        data_path=data_path,
        model_path=model_path,
    )
    return _make_output(signal)


# ---------------------------
# CLI entrypoint (unchanged)
# ---------------------------
def main(return_output: bool = False):
    args = parse_args()
    signal = generate_signal(
        timestamp=args.timestamp,
        data_path=args.data,
        model_path=args.model,
    )

    if return_output:
        return _make_output(signal)

    print_signal(signal, output_json=args.json)


if __name__ == "__main__":  # pragma: no cover
    main()
