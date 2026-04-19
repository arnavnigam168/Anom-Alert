from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

from config import (
    FEATURE_NAMES,
    FEATURE_NAMES_PATH,
    RANDOM_SEED,
    ROLLING_WINDOW,
    SCALER_PATH,
    SYNTHETIC_DATA_PATH,
    TIME_SERIES_STEPS,
)


def _rolling_variance_mean(signal: np.ndarray, window: int) -> float:
    if signal.size < window:
        return float(np.var(signal))
    vars_: List[float] = []
    for i in range(signal.size - window + 1):
        vars_.append(float(np.var(signal[i : i + window])))
    return float(np.mean(vars_)) if vars_ else float(np.var(signal))


def _series_features(signal: np.ndarray) -> Dict[str, float]:
    diffs = np.diff(signal)
    spike_threshold = 0.25  # for synthetic signals, distinguishes spikes vs stable jitter

    return {
        "ts_mean": float(np.mean(signal)),
        "ts_std": float(np.std(signal)),
        "ts_max": float(np.max(signal)),
        "ts_min": float(np.min(signal)),
        "ts_slope": float(np.mean(diffs) if diffs.size else 0.0),
        "ts_max_jump": float(np.max(np.abs(diffs)) if diffs.size else 0.0),
        "ts_rolling_var_mean": _rolling_variance_mean(signal, window=ROLLING_WINDOW),
        "ts_spike_count": float(np.sum(np.abs(diffs) > spike_threshold)),
    }


def _generate_time_series(
    rng: np.random.Generator,
    label: int,
    time_steps: int,
) -> Tuple[np.ndarray, float]:
    """
    Generates a synthetic "sensor trace" whose aggregated features drive the label:
    - PASS: stable, low noise, low drift
    - REVIEW: mild drift / borderline variability
    - FAIL: spikes + high variability
    """
    t = np.linspace(0, 1, time_steps)

    if label == 0:  # PASS
        base = rng.normal(1.0, 0.03)
        drift = rng.normal(0.0, 0.008)
        noise = rng.normal(0.05, 0.008)
        signal = base + drift * t + rng.normal(0.0, noise, size=time_steps)
        # Ensure stability: no large spikes
        signal = signal.astype(float)
        spike_amp = float(rng.normal(0.0, 0.03))
        return signal + spike_amp * 0.0, float(noise)

    if label == 2:  # REVIEW
        base = rng.normal(1.0, 0.05)
        drift = rng.normal(0.0, 0.02)
        noise = rng.normal(0.07, 0.015)
        signal = base + drift * t + rng.normal(0.0, noise, size=time_steps)

        # Borderline: sometimes one small spike
        if rng.random() < 0.35:
            idx = int(rng.integers(low=3, high=time_steps - 3))
            signal[idx] += float(rng.normal(0.18, 0.04))

        return signal.astype(float), float(noise)

    # label == 1: FAIL
    base = rng.normal(1.0, 0.07)
    drift = rng.normal(0.0, 0.05)
    noise = rng.normal(0.12, 0.03)
    signal = base + drift * t + rng.normal(0.0, noise, size=time_steps)

    # Spikes: add 2-3 large jumps
    n_spikes = int(rng.integers(low=2, high=4))
    spike_positions = rng.choice(np.arange(3, time_steps - 3), size=n_spikes, replace=False)
    for pos in spike_positions:
        signal[pos] += float(rng.normal(0.45, 0.12)) * rng.choice([1.0, -1.0])

    return signal.astype(float), float(noise)


def generate_synthetic_dataset(
    n_samples: int = 2400,
    time_steps: int = TIME_SERIES_STEPS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a balanced synthetic dataset for biologics batch QC:
    - PASS (0): stable values
    - REVIEW (2): mild drift / borderline values
    - FAIL (1): spikes, high temperature, high oxygen variance
    """
    rng = np.random.default_rng(random_seed)

    n_base = n_samples // 3
    remainder = n_samples % 3
    counts = {0: n_base, 1: n_base, 2: n_base}
    # Distribute remainder deterministically
    for cls in [0, 1, 2][:remainder]:
        counts[cls] += 1

    labels: List[int] = []
    for cls, count in counts.items():
        labels.extend([cls] * count)
    rng.shuffle(labels)

    rows: List[Dict[str, float]] = []

    for label in labels:
        # Batch-level signals that strongly drive label separation
        if label == 0:  # PASS
            pH_mean = float(rng.normal(7.0, 0.15))
            temp_max = float(np.clip(rng.normal(37.1, 0.55), 34.5, 38.2))
            oxygen_variance = float(np.clip(rng.gamma(shape=2.2, scale=0.035), 0.02, 0.22))
        elif label == 2:  # REVIEW
            pH_mean = float(rng.normal(7.1, 0.22))
            temp_max = float(np.clip(rng.normal(37.8, 0.65), 36.5, 38.4))
            oxygen_variance = float(np.clip(rng.gamma(shape=2.2, scale=0.055), 0.08, 0.35))
        else:  # label == 1 FAIL
            pH_mean = float(rng.normal(6.95, 0.22))
            temp_max = float(np.clip(rng.normal(39.1, 0.65), 38.3, 42.0))
            oxygen_variance = float(np.clip(rng.gamma(shape=2.2, scale=0.095), 0.22, 0.9))

        signal, _ = _generate_time_series(rng=rng, label=label, time_steps=time_steps)
        ts_feats = _series_features(signal)

        # Slight coupling to batch-level factors (keeps synthetic realism)
        # - Higher temp/oxygen variance tends to increase short-term variability
        ts_feats["ts_rolling_var_mean"] *= float(1.0 + 0.25 * (oxygen_variance - 0.15) + 0.08 * (temp_max - 37.5))
        ts_feats["ts_std"] *= float(1.0 + 0.18 * (oxygen_variance - 0.15) + 0.06 * (temp_max - 37.5))

        row: Dict[str, float] = {
            "pH_mean": pH_mean,
            "temp_max": temp_max,
            "oxygen_variance": oxygen_variance,
            **ts_feats,
            "label": float(label),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Label should be integer
    df["label"] = df["label"].astype(int)

    return df


def fit_scaler(df: pd.DataFrame, feature_names: List[str] = FEATURE_NAMES) -> StandardScaler:
    X = df[feature_names].values
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def save_preprocessing(
    scaler: StandardScaler,
    feature_names: List[str] = FEATURE_NAMES,
) -> None:
    FEATURE_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as fp:
        json.dump(feature_names, fp, indent=2)


def load_feature_names(path: str = str(FEATURE_NAMES_PATH)) -> List[str]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_scaler(path: str = str(SCALER_PATH)) -> StandardScaler:
    scaler_path = Path(path)
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
    obj = joblib.load(scaler_path)
    if not hasattr(obj, "transform"):
        raise TypeError(f"Expected a fitted scaler with transform(); got {type(obj).__name__}")
    return obj


def transform_input(
    payload: Union[Dict[str, float], Iterable[float], pd.DataFrame],
    feature_names: List[str],
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align raw input to training feature order, fill missing columns with 0, drop extras,
    then apply the training scaler.

    Returns:
        x_scaled: shape (1, n_features)
        x_unscaled: shape (n_features,) float64
    """
    names = list(feature_names)
    if not names:
        raise ValueError("feature_names is empty; cannot transform input.")

    if isinstance(payload, pd.DataFrame):
        if payload.shape[0] != 1:
            raise ValueError(f"Expected a single-row DataFrame; got {payload.shape[0]} rows.")
        frame = payload.reindex(columns=names, fill_value=0.0)
    elif isinstance(payload, dict):
        frame = pd.DataFrame([payload]).reindex(columns=names, fill_value=0.0)
    else:
        values = [float(v) for v in list(payload)]
        if len(values) != len(names):
            raise ValueError(f"Expected {len(names)} values in iterable payload, got {len(values)}.")
        frame = pd.DataFrame([values], columns=names)

    x_unscaled_2d = frame.to_numpy(dtype=np.float64)
    if np.isnan(x_unscaled_2d).any():
        bad = [names[i] for i in np.where(np.isnan(x_unscaled_2d[0]))[0].tolist()]
        raise ValueError(f"Input contains NaN after alignment (features: {bad}).")

    try:
        x_scaled = scaler.transform(x_unscaled_2d)
    except Exception as exc:
        logger.exception("Scaler transform failed")
        raise ValueError(f"Scaler transform failed: {exc}") from exc

    if np.isnan(x_scaled).any():
        raise ValueError("Scaled features contain NaN; check scaler parameters and input.")

    x_unscaled = x_unscaled_2d.reshape(-1)
    logger.debug("transform_input: x_unscaled shape=%s x_scaled shape=%s", x_unscaled.shape, x_scaled.shape)
    return x_scaled, x_unscaled


def ensure_synthetic_data(
    n_samples: int,
    time_steps: int,
    random_seed: int = RANDOM_SEED,
) -> Path:
    DATA_DIR = SYNTHETIC_DATA_PATH.parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC_DATA_PATH.exists():
        return SYNTHETIC_DATA_PATH
    df = generate_synthetic_dataset(
        n_samples=n_samples,
        time_steps=time_steps,
        random_seed=random_seed,
    )
    df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    return SYNTHETIC_DATA_PATH

