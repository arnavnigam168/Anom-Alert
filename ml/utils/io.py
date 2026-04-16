from __future__ import annotations

import joblib


def save_model(model, path) -> None:
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
