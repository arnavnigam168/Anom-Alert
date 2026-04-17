from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np

from config import LABEL_TO_NAME, MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH
from preprocess import load_feature_names, load_scaler
from utils.explain import predict_top_features_and_explanation
from utils.io import load_model


_MODEL = None
_SCALER = None
_FEATURE_NAMES: List[str] = []


def _artifact_path(value) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _load_artifacts() -> None:
    global _MODEL, _SCALER, _FEATURE_NAMES

    required = {
        "model.pkl": _artifact_path(MODEL_PATH),
        "scaler.pkl": _artifact_path(SCALER_PATH),
        "feature_names.json": _artifact_path(FEATURE_NAMES_PATH),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing ML artifact(s): "
            + ", ".join(missing)
            + ". Run `python train.py` inside /ml to regenerate artifacts."
        )

    if _MODEL is None:
        _MODEL = load_model(MODEL_PATH)
    if _SCALER is None:
        _SCALER = load_scaler(SCALER_PATH)
    if not _FEATURE_NAMES:
        _FEATURE_NAMES = load_feature_names(str(FEATURE_NAMES_PATH))


def _extract_prediction_and_confidence(model, x_scaled: np.ndarray) -> tuple[int, float, List[float]]:
    """
    Robust probability extraction for different model outputs.
    Handles edge-cases like:
    - predict_proba returning shape (1,) or scalar
    - single-class models
    - class index/label mismatches
    """
    classes = getattr(model, "classes_", None)
    fallback_label = int(classes[0]) if classes is not None and len(classes) > 0 else 0

    try:
        raw_proba = model.predict_proba(x_scaled)
    except Exception:
        # Fallback if model does not expose predict_proba reliably.
        pred_arr = np.asarray(model.predict(x_scaled)).reshape(-1)
        pred = int(pred_arr[0]) if pred_arr.size else fallback_label
        return pred, 1.0, [1.0]

    proba_arr = np.asarray(raw_proba)
    if proba_arr.size == 0:
        pred_arr = np.asarray(model.predict(x_scaled)).reshape(-1)
        pred = int(pred_arr[0]) if pred_arr.size else fallback_label
        return pred, 1.0, [1.0]

    # Normalize to a 1D "single sample" probability vector.
    if proba_arr.ndim == 0:
        one_row = np.array([float(proba_arr)], dtype=float)
    elif proba_arr.ndim == 1:
        one_row = proba_arr.astype(float)
    else:
        one_row = proba_arr[0].astype(float)

    one_row = np.nan_to_num(one_row, nan=0.0, posinf=1.0, neginf=0.0)

    # Single-value probability output edge case.
    if one_row.size == 1:
        if classes is not None and len(classes) >= 1:
            predicted_label = int(classes[0])
        else:
            predicted_label = fallback_label
        confidence = float(np.clip(one_row[0], 0.0, 1.0))
        return predicted_label, confidence, [confidence]

    # Multi-class probabilities.
    local_idx = int(np.argmax(one_row))
    if classes is not None and len(classes) > local_idx:
        predicted_label = int(classes[local_idx])
    else:
        predicted_label = local_idx

    confidence = float(np.clip(one_row[local_idx], 0.0, 1.0))
    return predicted_label, confidence, one_row.tolist()


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
        print("[ML][predict] input received:", payload)
        x_unscaled = np.array([float(payload.get(name, 0.0)) for name in _FEATURE_NAMES], dtype=float)
    else:
        values = list(payload)
        if len(values) != len(_FEATURE_NAMES):
            raise ValueError(f"Expected {len(_FEATURE_NAMES)} values, got {len(values)}.")
        x_unscaled = np.array([float(v) for v in values], dtype=float)
        print("[ML][predict] input received (array):", values)

    x_scaled = _SCALER.transform(x_unscaled.reshape(1, -1))
    print("[ML][predict] processed features:", dict(zip(_FEATURE_NAMES, x_unscaled.tolist())))
    predicted_label_int, confidence, proba_vector = _extract_prediction_and_confidence(_MODEL, x_scaled)
    print("[ML][predict] model probabilities:", proba_vector)

    # Safety fallback if model returns labels outside expected mapping.
    if predicted_label_int not in LABEL_TO_NAME:
        predicted_label_int = 2  # REVIEW as conservative fallback

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
        "confidence": round(float(confidence), 6),
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
