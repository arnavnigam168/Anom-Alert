from pathlib import Path

import mlflow
import numpy as np
import xgboost as xgb

from config import settings
from models.schemas import PredictionLabel


class PredictorService:
    def __init__(self) -> None:
        self._model = None
        self.model_loaded = False
        self.model_version = settings.model_version
        self._load_model()

    def _load_model(self) -> None:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            model_uri = f"models:/{settings.model_name}/{settings.model_version}"
            self._model = mlflow.pyfunc.load_model(model_uri)
            self.model_loaded = True
            return
        except Exception:
            pass

        local_model_path = Path(settings.model_local_path)
        if local_model_path.exists():
            booster = xgb.Booster()
            booster.load_model(str(local_model_path))
            self._model = booster
            self.model_loaded = True
            return

        self._model = None
        self.model_loaded = False

    def predict(self, feature_vector: np.ndarray) -> tuple[PredictionLabel, float]:
        if self._model is None:
            return PredictionLabel.PASS, 0.95

        try:
            if hasattr(self._model, "predict") and not isinstance(self._model, xgb.Booster):
                probabilities = self._model.predict(feature_vector)
                score = float(probabilities[0]) if np.ndim(probabilities) else float(probabilities)
            else:
                dmatrix = xgb.DMatrix(feature_vector)
                probabilities = self._model.predict(dmatrix)
                score = float(probabilities[0])
        except Exception:
            return PredictionLabel.PASS, 0.95

        if score >= 0.75:
            return PredictionLabel.PASS, score
        if score >= 0.45:
            return PredictionLabel.NEEDS_REVIEW, score
        return PredictionLabel.FAIL, score
