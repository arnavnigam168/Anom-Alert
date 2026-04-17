from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_NAMES,
    LABEL_TO_NAME,
    METRICS_PATH,
    MODEL_PATH,
    N_SAMPLES,
    RANDOM_SEED,
    TIME_SERIES_STEPS,
)
from preprocess import (
    ensure_synthetic_data,
    fit_scaler,
    generate_synthetic_dataset,
    save_preprocessing,
)
from utils.explain import mean_feature_importance


@dataclass
class CandidateResult:
    model_name: str
    model: object
    test_accuracy: float
    test_f1_macro: float
    cv_f1_macro_mean: float


def _evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    return acc, f1


def _cross_val_f1_macro(model, X: np.ndarray, y: np.ndarray, cv_splits: int = 5) -> float:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=None)
    return float(np.mean(scores))


def maybe_train_xgboost(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Optional[CandidateResult]:
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception:
        return None

   
    class_counts = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    n_total = float(len(y_train))
    class_weight = {cls: n_total / (3.0 * count) for cls, count in class_counts.items()}
    sample_weight = np.array([class_weight[int(v)] for v in y_train], dtype=float)

    xgb = XGBClassifier(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=0,
    )

    xgb.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    acc, f1 = _evaluate_classifier(xgb, X_test_scaled, y_test)
    return CandidateResult(
        model_name="XGBoost",
        model=xgb,
        test_accuracy=acc,
        test_f1_macro=f1,
        cv_f1_macro_mean=float("nan"),
    )


def train_model() -> Dict[str, object]:
    # Generate (or load) dataset
    ensure_synthetic_data(n_samples=N_SAMPLES, time_steps=TIME_SERIES_STEPS)
    df = generate_synthetic_dataset(n_samples=N_SAMPLES, time_steps=TIME_SERIES_STEPS)

    X = df[FEATURE_NAMES].values
    y = df["label"].values.astype(int)

    # Balanced split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Fit scaler on training only
    scaler = fit_scaler(
        # Fit scaler from training frame to keep feature names alignment.
        df=pd.DataFrame(X_train, columns=FEATURE_NAMES).assign(label=y_train),
        feature_names=FEATURE_NAMES,
    )
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate models
    # RF train
    rf = RandomForestClassifier(
        n_estimators=550,
        max_depth=13,
        min_samples_split=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    rf_acc, rf_f1 = _evaluate_classifier(rf, X_test_scaled, y_test)
    rf_cv_f1 = _cross_val_f1_macro(rf, X, y, cv_splits=5)

    print("Model: RandomForest")
    print(f"  CV mean f1_macro (5-fold): {rf_cv_f1:.4f}")
    print(f"  Test accuracy: {rf_acc:.4f}")
    print(f"  Test f1_macro: {rf_f1:.4f}")

    best = CandidateResult(
        model_name="RandomForest",
        model=rf,
        test_accuracy=rf_acc,
        test_f1_macro=rf_f1,
        cv_f1_macro_mean=rf_cv_f1,
    )

    # Optional XGBoost
    xgb_result = maybe_train_xgboost(X_train_scaled, X_test_scaled, y_train, y_test)
    if xgb_result is not None:
        print("Model: XGBoost")
        print(f"  Test accuracy: {xgb_result.test_accuracy:.4f}")
        print(f"  Test f1_macro: {xgb_result.test_f1_macro:.4f}")

        xgb_cv_f1 = _cross_val_f1_macro(xgb_result.model, X, y, cv_splits=5)
        xgb_result.cv_f1_macro_mean = xgb_cv_f1
        print(f"  CV mean f1_macro (5-fold): {xgb_cv_f1:.4f}")

        if xgb_result.test_f1_macro > best.test_f1_macro:
            best = xgb_result

    # SHAP feature importance summary (optional if shap isn't installed)
    try:
        shap_importance = mean_feature_importance(best.model, X_test_scaled, FEATURE_NAMES)
    except Exception:
        shap_importance = []

    # Save artifacts
    joblib.dump(best.model, MODEL_PATH)
    save_preprocessing(scaler=scaler, feature_names=FEATURE_NAMES)

    metrics = {
        "model_name": best.model_name,
        "accuracy": best.test_accuracy,
        "f1_macro": best.test_f1_macro,
        "cv_f1_macro_mean": best.cv_f1_macro_mean,
        "label_distribution": {LABEL_TO_NAME[int(k)]: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "feature_importance_shap": shap_importance[:15],
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_model()

