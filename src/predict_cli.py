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


def main() -> None:
    args = parse_args()
    signal = generate_signal(
        timestamp=args.timestamp,
        data_path=args.data,
        model_path=args.model,
    )
    print_signal(signal, output_json=args.json)


if __name__ == "__main__":  # pragma: no cover
    main()


