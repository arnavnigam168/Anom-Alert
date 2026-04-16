from __future__ import annotations

import json
from typing import Dict, Iterable, Union

import numpy as np

from config import LABEL_TO_NAME, MODEL_PATH
from preprocess import Preprocessor
from utils.explain import top_contributing_features
from utils.io import load_model


def predict(payload: Union[Dict[str, float], Iterable[float]]) -> Dict[str, object]:
    model = load_model(MODEL_PATH)
    preprocessor = Preprocessor.load()

    sample = preprocessor.transform_input(payload)
    probabilities = model.predict_proba(sample)[0]
    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])
    top_features = top_contributing_features(
        model=model,
        sample=sample,
        feature_names=preprocessor.feature_names,
        top_k=2,
    )

    return {
        "prediction": LABEL_TO_NAME[predicted_label],
        "confidence": round(confidence, 4),
        "top_features": top_features,
    }


if __name__ == "__main__":
    sample_input = {
        "pH_mean": 6.95,
        "temp_max": 37.9,
        "oxygen_variance": 0.16,
        "ts_mean": 1.03,
        "ts_std": 0.17,
        "ts_max": 1.31,
        "ts_min": 0.77,
    }
    print(json.dumps(predict(sample_input), indent=2))
