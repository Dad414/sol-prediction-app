
import sys
from pathlib import Path
import pandas as pd
import pickle

# Add src to path
sys.path.append(str(Path("src").resolve()))

from signal_generation import _load_model

def main():
    model_path = Path("models/sol_trend_random_forest.pkl")
    try:
        print(f"Attempting to load model from {model_path}...")
        data = _load_model(model_path)
        print("Model loaded successfully!")
        print("Keys:", data.keys())
        if "model" in data:
            print("Model type:", type(data["model"]))
            print("Model classes:", data["model"].classes_)
        if "features" in data:
            print("Features:", data["features"])
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()
