from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List, Union

import numpy as np

from config import FEATURE_NAMES, LABEL_TO_NAME, MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH
from preprocess import load_feature_names, load_scaler
from utils.explain import predict_top_features_and_explanation
from utils.io import load_model


_MODEL = None
_SCALER = None
_FEATURE_NAMES: List[str] = []


def _load_artifacts():
    global _MODEL, _SCALER, _FEATURE_NAMES
    if _MODEL is None:
        _MODEL = load_model(MODEL_PATH)
    if _SCALER is None:
        _SCALER = load_scaler(SCALER_PATH)
    if not _FEATURE_NAMES:
        _FEATURE_NAMES = load_feature_names(str(FEATURE_NAMES_PATH))


def predict(payload: Union[Dict[str, float], Iterable[float]]) -> Dict[str, object]:
    """
    Backend-compatible prediction function.

    Args:
        payload: either a dict of feature->value or an iterable with the
                 same ordering as `FEATURE_NAMES`.

    Returns:
        {
          "prediction": "PASS/FAIL/REVIEW",
          "confidence": float,
          "top_features": ["feature1", "feature2", "feature3"],
          "explanation": "Short human-readable reason"
        }
    """
    _load_artifacts()
    assert _MODEL is not None and _SCALER is not None

    if isinstance(payload, dict):
        x_unscaled = np.array([float(payload.get(name, 0.0)) for name in _FEATURE_NAMES], dtype=float)
    else:
        values = list(payload)
        if len(values) != len(_FEATURE_NAMES):
            raise ValueError(f"Expected {len(_FEATURE_NAMES)} values, got {len(values)}.")
        x_unscaled = np.array([float(v) for v in values], dtype=float)

    x_scaled = _SCALER.transform(x_unscaled.reshape(1, -1))
    proba = _MODEL.predict_proba(x_scaled)[0]
    predicted_label_int = int(np.argmax(proba))
    confidence = float(proba[predicted_label_int])

    top_features, explanation = predict_top_features_and_explanation(
        model=_MODEL,
        x_scaled=x_scaled,
        x_unscaled=x_unscaled,
        feature_names=_FEATURE_NAMES,
        predicted_label_int=predicted_label_int,
        top_k=3,
    )

    return {
        "prediction": LABEL_TO_NAME[predicted_label_int],
        "confidence": confidence,
        "top_features": top_features,
        "explanation": explanation,
    }


def _cli():
    parser = argparse.ArgumentParser(description="Anom Alert ML prediction")
    parser.add_argument(
        "--payload",
        type=str,
        default="",
        help="JSON dict with keys for features (feature->value).",
    )
    args = parser.parse_args()

    if args.payload:
        payload = json.loads(args.payload)
        print(json.dumps(predict(payload), indent=2))
    else:
        sample_input = {
            "pH_mean": 6.95,
            "temp_max": 37.9,
            "oxygen_variance": 0.16,
            "ts_mean": 1.03,
            "ts_std": 0.17,
            "ts_max": 1.31,
            "ts_min": 0.77,
            "ts_slope": 0.01,
            "ts_max_jump": 0.12,
            "ts_rolling_var_mean": 0.01,
            "ts_spike_count": 0,
        }
        print(json.dumps(predict(sample_input), indent=2))


if __name__ == "__main__":
    _cli()
