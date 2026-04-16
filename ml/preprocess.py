from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import PREPROCESSOR_PATH, RANDOM_SEED


def _series_stats(signal: np.ndarray) -> Dict[str, float]:
    return {
        "ts_mean": float(np.mean(signal)),
        "ts_std": float(np.std(signal)),
        "ts_max": float(np.max(signal)),
        "ts_min": float(np.min(signal)),
    }


def generate_synthetic_dataset(
    n_samples: int = 1500,
    time_steps: int = 24,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, float]] = []

    for _ in range(n_samples):
        pH_mean = float(rng.normal(7.0, 0.25))
        temp_max = float(rng.normal(37.0, 2.0))
        oxygen_variance = float(np.clip(rng.gamma(shape=2.0, scale=0.07), 0.001, None))

        base_signal = rng.normal(loc=1.0, scale=0.15, size=time_steps)
        drift = rng.normal(0.0, 0.02)
        trend = np.linspace(0.0, drift * time_steps, time_steps)
        signal = base_signal + trend + rng.normal(0.0, 0.02, size=time_steps)
        stats = _series_stats(signal)

        risk_score = (
            abs(pH_mean - 7.0) * 2.8
            + max(0.0, temp_max - 38.0) * 0.85
            + max(0.0, 35.0 - temp_max) * 0.35
            + oxygen_variance * 9.0
            + abs(stats["ts_std"] - 0.12) * 6.0
            + max(0.0, stats["ts_max"] - 1.3) * 2.0
            + max(0.0, 0.7 - stats["ts_min"]) * 2.2
        )
        risk_score += float(rng.normal(0.0, 0.18))

        if risk_score >= 2.0:
            label = 1  # FAIL
        elif risk_score >= 1.1:
            label = 2  # REVIEW
        else:
            label = 0  # PASS

        rows.append(
            {
                "pH_mean": pH_mean,
                "temp_max": temp_max,
                "oxygen_variance": oxygen_variance,
                **stats,
                "label": label,
            }
        )

    return pd.DataFrame(rows)


@dataclass
class Preprocessor:
    scaler: StandardScaler
    feature_names: List[str]

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        # Use numpy arrays to avoid sklearn feature-name warnings across versions.
        return self.scaler.transform(df[self.feature_names].values)

    def transform_input(self, payload: Union[Dict[str, float], Iterable[float]]) -> np.ndarray:
        if isinstance(payload, dict):
            row = pd.DataFrame([
                {feature: payload.get(feature, 0.0) for feature in self.feature_names}
            ])
        else:
            values = list(payload)
            if len(values) != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} values, got {len(values)}."
                )
            row = pd.DataFrame([values], columns=self.feature_names)
        return self.scaler.transform(row.values)

    def save(self, path=PREPROCESSOR_PATH) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path=PREPROCESSOR_PATH) -> "Preprocessor":
        import joblib
        return joblib.load(path)


def fit_preprocessor(df: pd.DataFrame, target_column: str = "label") -> tuple[np.ndarray, np.ndarray, Preprocessor]:
    feature_names = [col for col in df.columns if col != target_column]
    X = df[feature_names].values
    y = df[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    preprocessor = Preprocessor(scaler=scaler, feature_names=feature_names)
    return X_scaled, y, preprocessor
