from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from config import LABEL_TO_NAME

logger = logging.getLogger(__name__)


def _safe_get_shap():
    try:
        import shap  # type: ignore

        return shap
    except ModuleNotFoundError:
        return None


def _top_features_from_importances(
    model, x_scaled: np.ndarray, feature_names: List[str], top_k: int
) -> List[str]:
    # Tree models expose global feature importances; we combine with per-sample magnitude.
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        # Fallback: choose largest standardized absolute values.
        idx = np.argsort(np.abs(x_scaled[0]))[::-1][:top_k]
        return [feature_names[int(i)] for i in idx]

    importances = np.asarray(importances, dtype=float)
    per_feature = np.abs(x_scaled[0]) * importances
    idx = np.argsort(per_feature)[::-1][:top_k]
    return [feature_names[int(i)] for i in idx]


def top_contributing_features(
    model,
    x_scaled: np.ndarray,
    feature_names: List[str],
    predicted_label_int: int,
    top_k: int = 3,
) -> List[str]:
    shap = _safe_get_shap()
    if shap is None:
        return _top_features_from_importances(model, x_scaled, feature_names, top_k=top_k)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_scaled)
    except Exception as exc:
        logger.warning("SHAP TreeExplainer failed (%s); using feature importance fallback.", exc)
        return _top_features_from_importances(model, x_scaled, feature_names, top_k=top_k)

    # Multiclass SHAP is commonly a list[class] -> [sample, features]
    if isinstance(shap_values, list):
        class_idx = int(predicted_label_int)
        if class_idx < 0 or class_idx >= len(shap_values):
            class_idx = int(np.clip(class_idx, 0, len(shap_values) - 1))
        contrib = np.abs(shap_values[class_idx][0])
    else:
        values = np.array(shap_values)
        # Handle shapes like (n_samples, n_features) or (n_classes, n_samples, n_features)
        if values.ndim == 3:
            class_idx = int(predicted_label_int)
            class_idx = int(np.clip(class_idx, 0, values.shape[0] - 1))
            contrib = np.abs(values[class_idx, 0, :])
        else:
            contrib = np.abs(values[0])

    idx = np.argsort(contrib)[::-1][:top_k]
    return [feature_names[int(i)] for i in idx]


def _feature_reason(feature: str, value: float, predicted_label: str) -> str:
    if feature == "oxygen_variance":
        return f"Oxygen variance is {value:.3f}, indicating instability consistent with {predicted_label}."
    if feature == "temp_max":
        return f"Max temperature is {value:.2f}, suggesting elevated stress conditions."
    if feature == "pH_mean":
        return f"pH_mean is {value:.3f}, deviating from the stable target range."
    if feature == "ts_slope":
        return f"Time-series drift (slope) is {value:.3f}, suggesting trend instability."
    if feature == "ts_max_jump":
        return f"Largest time-series jump is {value:.3f}, indicating abrupt disturbances."
    if feature == "ts_rolling_var_mean":
        return f"Short-term variability (rolling var) is {value:.4f}, pointing to inconsistent behavior."
    if feature == "ts_spike_count":
        c = int(value)
        if c <= 0:
            return "Spike count is 0, indicating a relatively stable trace with few abrupt changes."
        if c <= 1:
            return f"Spike count is {c}, indicating occasional spikes in the trace."
        return f"Spike count is {c}, indicating frequent spikes and instability in the trace."
    if feature == "ts_std":
        return f"Time-series standard deviation is {value:.4f}, showing elevated variability."
    if feature == "ts_max":
        return f"Time-series maximum is {value:.3f}, reflecting an extreme excursion."
    if feature == "ts_min":
        return f"Time-series minimum is {value:.3f}, reflecting a low excursion."
    if feature == "ts_mean":
        return f"Time-series mean is {value:.3f}, differing from typical stable patterns."
    return f"Feature {feature} contributes to the {predicted_label} decision."


def predict_top_features_and_explanation(
    model,
    x_scaled: np.ndarray,
    x_unscaled: np.ndarray,
    feature_names: List[str],
    predicted_label_int: int,
    top_k: int = 3,
) -> Tuple[List[str], str]:
    predicted_label = LABEL_TO_NAME.get(int(predicted_label_int), "REVIEW")

    top_features = top_contributing_features(
        model=model,
        x_scaled=x_scaled,
        feature_names=feature_names,
        predicted_label_int=predicted_label_int,
        top_k=top_k,
    )

    x_unscaled_flat = np.asarray(x_unscaled).reshape(-1)
    x_unscaled_map: Dict[str, float] = {name: float(val) for name, val in zip(feature_names, x_unscaled_flat)}
    reasons: List[str] = []
    for feat in top_features:
        value = x_unscaled_map.get(feat, float("nan"))
        reasons.append(_feature_reason(feat, value, predicted_label))

    explanation = "Because " + "; ".join(reasons) + "."
    return top_features, explanation


def mean_feature_importance(
    model, X_scaled: np.ndarray, feature_names: List[str], top_k: int = 10
) -> List[dict]:
    """
    Returns a ranked list of mean absolute feature contributions.
    Uses SHAP if available, otherwise falls back to model.feature_importances_.
    """
    shap = _safe_get_shap()
    if shap is None:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return []
        importances = np.asarray(importances, dtype=float)
        idx = np.argsort(importances)[::-1][:top_k]
        return [
            {"feature": feature_names[int(i)], "importance": float(importances[int(i)])}
            for i in idx
        ]

    # SHAP can be slow; use a modest subset for summary.
    X_sub = X_scaled[: min(250, X_scaled.shape[0])]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    if isinstance(shap_values, list):
        # Average abs contributions across classes and samples
        stacked = np.stack([np.abs(v) for v in shap_values], axis=0)  # [classes, samples, features]
        importances = stacked.mean(axis=(0, 1))
    else:
        values = np.array(shap_values)
        if values.ndim == 3:
            # [classes, samples, features]
            importances = np.abs(values).mean(axis=(0, 1))
        else:
            importances = np.abs(values).mean(axis=0)

    idx = np.argsort(importances)[::-1][:top_k]
    return [
        {"feature": feature_names[int(i)], "importance": float(importances[int(i)])}
        for i in idx
    ]
