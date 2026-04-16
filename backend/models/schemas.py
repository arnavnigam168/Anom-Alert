from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PredictionLabel(str, Enum):
    PASS = "PASS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAIL = "FAIL"


class SensorPoint(BaseModel):
    timestamp: datetime
    ph: float | None = None
    dissolved_oxygen: float | None = None
    temperature: float | None = None

    model_config = {"extra": "allow"}


class LabAssays(BaseModel):
    cell_viability: float
    titer_g_per_l: float
    endotoxin_eu_per_ml: float
    osmolality: float
    aggregation_pct: float

    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    batch_id: str = Field(..., min_length=1)
    sensor_data: list[SensorPoint]
    lab_assays: LabAssays


class ShapFeature(BaseModel):
    feature: str
    impact: float
    direction: str


class ShapExplanation(BaseModel):
    top_features: list[ShapFeature]
    summary_text: str


class AuditInfo(BaseModel):
    decision_id: UUID
    timestamp: datetime
    model_version: str


class PredictResponse(BaseModel):
    batch_id: str
    prediction: PredictionLabel
    confidence: float
    shap_explanation: ShapExplanation
    audit: AuditInfo


class BatchSummaryResponse(BaseModel):
    decision_id: UUID
    batch_id: str
    prediction: PredictionLabel
    confidence: float
    model_version: str
    created_at: datetime


class BatchDetailResponse(BatchSummaryResponse):
    shap_explanation: dict[str, Any]
    input_payload: dict[str, Any]
    signed_by: str
