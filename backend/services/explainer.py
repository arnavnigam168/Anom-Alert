import numpy as np
import shap

from models.schemas import PredictionLabel


class ExplainerService:
    def explain(
        self,
        model,
        feature_vector: np.ndarray,
        feature_names: list[str],
        feature_map: dict,
        prediction: PredictionLabel,
    ) -> dict:
        top_features = self._compute_top_features(model, feature_vector, feature_names, feature_map)
        summary_text = self._summary_text(prediction, top_features)
        return {"top_features": top_features, "summary_text": summary_text}

    def _compute_top_features(
        self,
        model,
        feature_vector: np.ndarray,
        feature_names: list[str],
        feature_map: dict,
    ) -> list[dict]:
        if model is None:
            return self._fallback_features(feature_map)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(feature_vector)
            values = shap_values[0] if isinstance(shap_values, np.ndarray) else np.array(shap_values)[0]

            pairs = []
            for idx, value in enumerate(values):
                feature = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                pairs.append((feature, float(value)))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            top = pairs[:5]
            return [
                {
                    "feature": feature,
                    "impact": impact,
                    "direction": "positive" if impact >= 0 else "negative",
                }
                for feature, impact in top
            ]
        except Exception:
            return self._fallback_features(feature_map)

    def _fallback_features(self, feature_map: dict) -> list[dict]:
        ranked = sorted(feature_map.items(), key=lambda x: abs(float(x[1])), reverse=True)[:5]
        return [
            {
                "feature": name,
                "impact": round(float(value) / 10.0, 3),
                "direction": "positive" if float(value) >= 0 else "negative",
            }
            for name, value in ranked
        ]

    def _summary_text(self, prediction: PredictionLabel, top_features: list[dict]) -> str:
        if not top_features:
            return "Prediction generated with default fallback explanation."

        lead = ", ".join(feature["feature"] for feature in top_features[:2])
        if prediction == PredictionLabel.PASS:
            return f"Batch passed primarily due to favorable signals in {lead}."
        if prediction == PredictionLabel.NEEDS_REVIEW:
            return f"Batch needs review due to mixed signals led by {lead}."
        return f"Batch failed due to adverse feature impact primarily from {lead}."
