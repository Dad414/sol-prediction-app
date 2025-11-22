"""Utilities for downloading SOL/USDT 30 minute klines from Binance.

This script supports incremental updates of a persisted CSV located at
`data/raw/solusdt_30m.csv`. A historical backfill can be performed by supplying
an explicit `--start` datetime while leaving `--end` blank to pull up through
the latest available candle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
SYMBOL = "SOLUSDT"
INTERVAL = "30m"
MAX_LIMIT = 1000
OUTPUT_PATH = Path("data/raw/solusdt_30m.csv")

logger = logging.getLogger(__name__)


def _parse_human_datetime(value: str) -> dt.datetime:
    """Parse a human friendly datetime string (ISO8601 variants)."""
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"Invalid datetime string: {value}") from exc


def _epoch_ms(dt_obj: dt.datetime) -> int:
    """Convert aware datetime to milliseconds since epoch."""
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    return int(dt_obj.timestamp() * 1000)


def _klines_request(start_ms: int, end_ms: Optional[int] = None) -> List[List[float]]:
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": MAX_LIMIT,
        "startTime": start_ms,
    }
    if end_ms is not None:
        params["endTime"] = end_ms

    response = requests.get(f"{BINANCE_BASE_URL}{KLINES_ENDPOINT}", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        msg = f"Unexpected response payload: {data}"
        raise RuntimeError(msg)
    return data


def _klines_to_df(rows: Iterable[List[float]]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop(columns=["ignore"])
    return df


def load_existing(path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["open_time", "close_time"])
    return pd.DataFrame()


def save_dataframe(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("open_time").drop_duplicates(subset="open_time").to_csv(path, index=False)
    logger.info("Persisted %s rows to %s", len(df), path)


def fetch_history(start: Optional[dt.datetime], end: Optional[dt.datetime]) -> pd.DataFrame:
    """Iteratively fetch klines between start and end (inclusive)."""
    if end is None:
        end = dt.datetime.now(tz=dt.timezone.utc)
    start_ms = _epoch_ms(start) if start else None
    end_ms = _epoch_ms(end)

    all_rows: List[List[float]] = []
    next_start = start_ms
    while True:
        params_start = next_start or (end_ms - MAX_LIMIT * 30 * 60 * 1000)
        chunk = _klines_request(start_ms=params_start, end_ms=end_ms)
        if not chunk:
            break

        all_rows.extend(chunk)

        last_close = chunk[-1][6]
        if next_start is not None and last_close <= next_start:
            break  # Avoid infinite loop if Binance returns overlapping data
        next_start = last_close + 1

        if len(chunk) < MAX_LIMIT:
            break

        time.sleep(0.25)  # Respect Binance API rate limits

    return _klines_to_df(all_rows)


def update_history(
    output_path: Path = OUTPUT_PATH,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    existing = load_existing(output_path)
    if not existing.empty:
        last_timestamp = existing["open_time"].max()
        start = max(start or last_timestamp, last_timestamp)
        start = start + dt.timedelta(minutes=30)
    elif start is None:
        start = (dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=90)).replace(minute=0, second=0, microsecond=0)

    logger.info("Fetching klines from %s to %s", start, end)
    new_data = fetch_history(start=start, end=end)
    if new_data.empty:
        logger.info("No new data retrieved.")
        return existing

    combined = pd.concat([existing, new_data], axis=0, ignore_index=True)
    save_dataframe(combined, output_path)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SOL/USDT 30m klines from Binance.")
    parser.add_argument("--start", type=_parse_human_datetime, help="Start datetime (ISO8601).")
    parser.add_argument("--end", type=_parse_human_datetime, help="End datetime (ISO8601).")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output CSV path (default: data/raw/solusdt_30m.csv).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    update_history(output_path=args.output, start=args.start, end=args.end)


if __name__ == "__main__":  # pragma: no cover
    main()



