from __future__ import annotations

from typing import List

import numpy as np


def top_contributing_features(model, sample: np.ndarray, feature_names: List[str], top_k: int = 2) -> List[str]:
    """
    Returns a small list of the most influential features for a single sample.

    If `shap` is installed, this uses SHAP values (TreeExplainer).
    If not, it falls back to a simple heuristic based on global feature importance
    multiplied by the (absolute) standardized feature value.
    """
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            class_idx = int(np.argmax(model.predict_proba(sample)[0]))
            per_feature = np.abs(shap_values[class_idx][0])
        else:
            values = np.array(shap_values)
            if values.ndim == 3:
                class_idx = int(np.argmax(model.predict_proba(sample)[0]))
                per_feature = np.abs(values[class_idx, 0, :])
            else:
                per_feature = np.abs(values[0])
    except ModuleNotFoundError:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise RuntimeError(
                "SHAP is not installed and the model does not expose feature_importances_. "
                "Install shap or use a tree-based model."
            )
        per_feature = np.abs(sample[0]) * np.asarray(importances)

    top_idx = np.argsort(per_feature)[::-1][:top_k]
    return [feature_names[int(i)] for i in top_idx]


def mean_feature_importance(model, X: np.ndarray, feature_names: List[str]) -> List[dict]:
    """
    Returns a ranked list of mean absolute feature importances.

    If `shap` is installed, uses mean(|SHAP|).
    Otherwise falls back to model.feature_importances_ (tree models).
    """
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            stacked = np.stack([np.abs(v) for v in shap_values], axis=0)
            importances = stacked.mean(axis=(0, 1))
        else:
            values = np.array(shap_values)
            if values.ndim == 3:
                importances = np.abs(values).mean(axis=(0, 1))
            else:
                importances = np.abs(values).mean(axis=0)
    except ModuleNotFoundError:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise RuntimeError(
                "SHAP is not installed and the model does not expose feature_importances_. "
                "Install shap or use a tree-based model."
            )
        importances = np.asarray(importances, dtype=float)

    ranking = np.argsort(importances)[::-1]
    return [
        {"feature": feature_names[int(idx)], "importance": float(importances[int(idx)])}
        for idx in ranking
    ]
