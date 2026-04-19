from __future__ import annotations

from pathlib import Path

import joblib


def save_model(model, path) -> None:
    joblib.dump(model, path)


def load_model(path):
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)
