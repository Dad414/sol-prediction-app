"""Train an XGBoost classifier for SOL/USDT trend prediction."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

from feature_engineering import compute_features
from labeling import generate_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/raw/solusdt_30m.csv")
DEFAULT_MODEL_PATH = Path("models/sol_trend_xgb.json")  # Save as JSON for XGBoost
DEFAULT_METRICS_PATH = Path("reports/model_metrics_xgb.json")


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
    """Train XGBoost model with TimeSeriesSplit validation and hyperparameter tuning."""
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

    # Encode labels (long, short, neutral) to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Map back for readability if needed, but XGBoost needs ints
    # 0: long, 1: neutral, 2: short (example, depends on alphabetical order)
    logger.info(f"Classes: {le.classes_}")

    logger.info(f"Training on {len(df)} samples with {len(feature_cols)} features.")

    # TimeSeriesSplit validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []

    logger.info(f"Running {n_splits}-fold TimeSeriesSplit validation...")
    
    # Hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]

        # Use RandomizedSearchCV on the first fold to find good params, then reuse?
        # Or just train a fixed good config to save time?
        # Let's do a quick search on each fold or just a fixed robust config.
        # For speed in this context, let's use a robust fixed config but slightly tuned.
        # Actually, let's do RandomizedSearchCV on the LAST fold's training data (largest) 
        # to find best params for the final model, but for CV loop, use a standard config to save time.
        
        clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            # class_weight is not directly supported in XGBClassifier for multi-class in the same way as sklearn
            # We can handle imbalance via sample_weight if needed, but let's start without.
        )
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        
        # Calculate metrics for this fold
        # Note: y_val and y_pred are integers now
        fold_metrics = {
            "fold": fold + 1,
            "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
            "accuracy": clf.score(X_val, y_val)
        }
        metrics.append(fold_metrics)
        logger.info(f"Fold {fold + 1} - F1 (Macro): {fold_metrics['f1_macro']:.4f}")

    # Train final model on all data with hyperparameter tuning
    logger.info("Tuning and training final model on all data...")
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_jobs=-1
    )
    
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist, 
        n_iter=20, 
        scoring='f1_macro', 
        n_jobs=-1, 
        cv=TimeSeriesSplit(n_splits=3), # Internal CV for tuning
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X, y_encoded)
    
    best_model = random_search.best_estimator_
    logger.info(f"Best parameters: {random_search.best_params_}")

    # Save model artifact
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save XGBoost model
    best_model.save_model(model_path)
    
    # Save metadata (features, encoder) separately
    metadata = {
        "features": feature_cols,
        "classes": le.classes_.tolist(),
        "timestamp": pd.Timestamp.now().isoformat(),
        "best_params": random_search.best_params_
    }
    metadata_path = model_path.with_suffix('.metadata.pkl')
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
        
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")

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
    
    print("\nAverage Validation Metrics (Baseline Config):")
    print(f"Accuracy: {avg_metrics['accuracy']:.2%}")
    print(f"F1 Score (Macro): {avg_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOL/USDT trend prediction model with XGBoost.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to raw data CSV.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to save model artifact.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON.")
    args = parser.parse_args()

    train_model(args.data, args.model, args.metrics)
