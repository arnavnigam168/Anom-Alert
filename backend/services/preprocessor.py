import numpy as np
import pandas as pd
from tsfresh import extract_features


SPEC_RANGES = {
    "cell_viability": (85.0, 100.0),
    "titer_g_per_l": (1.0, 6.0),
    "endotoxin_eu_per_ml": (0.0, 0.5),
    "osmolality": (260.0, 340.0),
    "aggregation_pct": (0.0, 5.0),
}


class Preprocessor:
    def preprocess(self, sensor_data: list[dict], lab_assays: dict) -> tuple[np.ndarray, list[str], dict]:
        ts_features, ts_names = self._extract_time_series_features(sensor_data)
        lab_features, lab_names = self._normalize_lab_assays(lab_assays)

        feature_vector = np.concatenate([ts_features, lab_features], dtype=float)
        feature_names = ts_names + lab_names
        feature_map = dict(zip(feature_names, feature_vector.tolist(), strict=False))
        return feature_vector.reshape(1, -1), feature_names, feature_map

    def _extract_time_series_features(self, sensor_data: list[dict]) -> tuple[np.ndarray, list[str]]:
        if not sensor_data:
            default_names = ["ph_mean", "dissolved_oxygen_mean", "temperature_mean"]
            return np.array([0.0, 0.0, 0.0], dtype=float), default_names

        frame = pd.DataFrame(sensor_data)
        frame["id"] = 1
        frame["time"] = pd.to_datetime(frame["timestamp"])
        value_columns = [col for col in frame.columns if col not in {"timestamp", "id", "time"}]

        if value_columns:
            tsfresh_frame = frame[["id", "time", *value_columns]].copy()
            ts = extract_features(tsfresh_frame, column_id="id", column_sort="time", disable_progressbar=True)
            ts = ts.fillna(0.0)
            feature_names = list(ts.columns)
            values = ts.iloc[0].to_numpy(dtype=float)
            if len(values) > 100:
                values = values[:100]
                feature_names = feature_names[:100]
            return values, feature_names

        default_names = ["ph_mean", "dissolved_oxygen_mean", "temperature_mean"]
        return np.array([0.0, 0.0, 0.0], dtype=float), default_names

    def _normalize_lab_assays(self, lab_assays: dict) -> tuple[np.ndarray, list[str]]:
        normalized_values: list[float] = []
        feature_names: list[str] = []
        for key, value in lab_assays.items():
            low, high = SPEC_RANGES.get(key, (0.0, 1.0))
            denominator = high - low if high > low else 1.0
            normalized = float((float(value) - low) / denominator)
            normalized_values.append(normalized)
            feature_names.append(key)

        return np.array(normalized_values, dtype=float), feature_names
