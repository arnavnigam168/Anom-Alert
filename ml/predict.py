from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np

from config import LABEL_TO_NAME, MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH
from preprocess import load_feature_names, load_scaler, transform_input
from utils.explain import predict_top_features_and_explanation
from utils.io import load_model

logger = logging.getLogger(__name__)

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
        _SCALER = load_scaler(str(SCALER_PATH))
    if not _FEATURE_NAMES:
        _FEATURE_NAMES = load_feature_names(str(FEATURE_NAMES_PATH))


def warm_artifacts() -> None:
    """Pre-load model, scaler, and feature list (e.g. FastAPI startup)."""
    _load_artifacts()


def _extract_prediction_and_confidence(model, x_scaled: np.ndarray) -> tuple[int, float, List[float]]:
    """
    Call predict_proba; if shape is (1, n_classes) use row [0].
    If shape is (1,) or the model has only one class, fall back to predict() with confidence 1.0.
    """
    classes = getattr(model, "classes_", None)
    n_classes = int(len(classes)) if classes is not None else 0

    def from_predict() -> tuple[int, float, List[float]]:
        pred_arr = np.asarray(model.predict(x_scaled)).reshape(-1)
        pred = int(pred_arr[0]) if pred_arr.size else 0
        return pred, 1.0, [1.0]

    if n_classes <= 1:
        return from_predict()

    try:
        raw_proba = model.predict_proba(x_scaled)
    except Exception as exc:
        logger.warning("predict_proba failed (%s); using predict() fallback.", exc)
        return from_predict()

    proba = np.asarray(raw_proba, dtype=float)

    if proba.ndim == 0:
        return from_predict()

    if proba.ndim == 1:
        if proba.shape[0] == 1:
            return from_predict()
        one_row = proba
    elif proba.ndim == 2:
        if proba.shape[0] < 1:
            return from_predict()
        if proba.shape[0] > 1:
            logger.warning(
                "predict_proba returned %s rows; using first row only.",
                proba.shape[0],
            )
        one_row = proba[0]
        if proba.shape == (1, 1) or one_row.size == 1:
            return from_predict()
    else:
        logger.warning("Unexpected predict_proba ndim=%s; using predict() fallback.", proba.ndim)
        return from_predict()

    if np.isnan(one_row).any():
        raise ValueError("Model returned NaN probabilities.")

    one_row = np.nan_to_num(one_row, nan=0.0, posinf=1.0, neginf=0.0)

    local_idx = int(np.argmax(one_row))
    if classes is not None and len(classes) > local_idx:
        predicted_label = int(classes[local_idx])
    else:
        predicted_label = local_idx

    confidence = float(np.clip(one_row[local_idx], 0.0, 1.0))
    return predicted_label, confidence, one_row.tolist()


def _fail_safe_response(message: str) -> Dict[str, object]:
    return {
        "prediction": "ERROR",
        "confidence": 0.0,
        "top_features": [],
        "explanation": message,
    }


def predict(payload: Union[Dict[str, float], Iterable[float]]) -> Dict[str, object]:
    """
    Backend-compatible prediction function.

    Returns:
        prediction, confidence, top_features, and explanation.
        On any failure, returns a fail-safe dict (prediction ERROR) without raising.
    """
    try:
        _load_artifacts()
    except Exception as exc:
        logger.exception("Artifact load failed: %s", exc)
        return _fail_safe_response(f"ML artifacts unavailable: {exc}")

    assert _MODEL is not None and _SCALER is not None

    try:
        work_payload: Union[Dict[str, float], Iterable[float]] = payload
        if isinstance(payload, dict):
            logger.info("[ML][predict] incoming payload keys=%s", sorted(payload.keys()))
        else:
            work_payload = [float(v) for v in list(payload)]
            logger.info("[ML][predict] incoming iterable length=%s", len(work_payload))

        x_scaled, x_unscaled = transform_input(
            payload=work_payload,
            feature_names=_FEATURE_NAMES,
            scaler=_SCALER,
        )
        logger.info(
            "[ML][predict] processed features shape=%s (scaled %s)",
            x_unscaled.shape,
            x_scaled.shape,
        )

        predicted_label_int, confidence, proba_vector = _extract_prediction_and_confidence(
            _MODEL, x_scaled
        )
        logger.info(
            "[ML][predict] model probabilities (summary)=%s final_label_int=%s confidence=%s",
            proba_vector,
            predicted_label_int,
            confidence,
        )

        if predicted_label_int not in LABEL_TO_NAME:
            logger.warning(
                "Predicted label %s not in LABEL_TO_NAME; mapping to REVIEW.",
                predicted_label_int,
            )
            predicted_label_int = 2

        top_features: List[str] = []
        explanation = ""
        try:
            top_features, explanation = predict_top_features_and_explanation(
                model=_MODEL,
                x_scaled=x_scaled,
                x_unscaled=x_unscaled,
                feature_names=_FEATURE_NAMES,
                predicted_label_int=predicted_label_int,
                top_k=3,
            )
        except Exception as exc:
            logger.warning("Feature explanation failed: %s", exc)
            explanation = "Explanation unavailable for this prediction."

        out = {
            "prediction": LABEL_TO_NAME[predicted_label_int],
            "confidence": round(float(confidence), 6),
            "top_features": top_features,
            "explanation": explanation,
        }
        logger.info(
            "[ML][predict] final prediction=%s confidence=%s",
            out["prediction"],
            out["confidence"],
        )
        return out

    except Exception as exc:
        logger.exception("Prediction pipeline failed: %s", exc)
        return _fail_safe_response(f"Inference failed: {exc}")


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


def _try_warm_at_import() -> None:
    try:
        _load_artifacts()
        logger.info("ML artifacts pre-loaded at module import.")
    except Exception as exc:
        logger.warning("ML artifacts not ready at import; will load on first predict: %s", exc)


_try_warm_at_import()

if __name__ == "__main__":
    _cli()
