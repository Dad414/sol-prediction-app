"""Train a Random Forest classifier for SOL/USDT trend prediction."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import compute_features
from labeling import generate_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/raw/solusdt_30m.csv")
DEFAULT_MODEL_PATH = Path("models/sol_trend_random_forest.pkl")
DEFAULT_METRICS_PATH = Path("reports/model_metrics.json")


def load_and_process_data(data_path: Path) -> pd.DataFrame:
    """Load raw data and generate features and labels."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["open_time", "close_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    logger.info("Computing features...")
    df = compute_features(df)

    logger.info("Generating labels...")
    df = generate_labels(df)
    
    # Drop rows with NaN values (created by lag features and labeling)
    df = df.dropna().reset_index(drop=True)
    return df


def train_model(
    data_path: Path,
    model_path: Path,
    metrics_path: Path,
    n_splits: int = 5,
) -> None:
    """Train model with TimeSeriesSplit validation."""
    df = load_and_process_data(data_path)

    # Define features and target
    feature_cols = [
        c for c in df.columns
        if c not in [
            "open_time", "close_time", "label", "future_close", "future_return",
            "open", "high", "low", "close", "volume", "quote_asset_volume",
            "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ]
    ]
    target_col = "label"

    X = df[feature_cols]
    y = df[target_col]

    logger.info(f"Training on {len(df)} samples with {len(feature_cols)} features.")
    logger.info(f"Features: {feature_cols}")

    # TimeSeriesSplit validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []

    logger.info(f"Running {n_splits}-fold TimeSeriesSplit validation...")
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        
        # Calculate metrics for this fold
        fold_metrics = {
            "fold": fold + 1,
            "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
            "accuracy": clf.score(X_val, y_val)
        }
        metrics.append(fold_metrics)
        logger.info(f"Fold {fold + 1} - F1 (Macro): {fold_metrics['f1_macro']:.4f}")

    # Train final model on all data
    logger.info("Training final model on all data...")
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    final_model.fit(X, y)

    # Save model artifact
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "features": feature_cols,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    avg_metrics = {
        "precision_macro": np.mean([m["precision_macro"] for m in metrics]),
        "recall_macro": np.mean([m["recall_macro"] for m in metrics]),
        "f1_macro": np.mean([m["f1_macro"] for m in metrics]),
        "accuracy": np.mean([m["accuracy"] for m in metrics]),
        "folds": metrics
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    print("\nAverage Validation Metrics:")
    print(f"Accuracy: {avg_metrics['accuracy']:.2%}")
    print(f"F1 Score (Macro): {avg_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOL/USDT trend prediction model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to raw data CSV.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to save model artifact.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON.")
    args = parser.parse_args()

    train_model(args.data, args.model, args.metrics)
