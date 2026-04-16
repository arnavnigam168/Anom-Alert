from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.audit_log import AuditLog, PredictionEnum
from db.database import get_db_session
from models.schemas import (
    BatchDetailResponse,
    BatchSummaryResponse,
    PredictRequest,
    PredictResponse,
    PredictionLabel,
)
from services.explainer import ExplainerService
from services.predictor import PredictorService
from services.preprocessor import Preprocessor

router = APIRouter()
preprocessor = Preprocessor()
predictor = PredictorService()
explainer = ExplainerService()


@router.post("/predict", response_model=PredictResponse)
async def predict_batch(payload: PredictRequest, db: AsyncSession = Depends(get_db_session)) -> PredictResponse:
    try:
        feature_vector, feature_names, feature_map = preprocessor.preprocess(
            sensor_data=[point.model_dump() for point in payload.sensor_data],
            lab_assays=payload.lab_assays.model_dump(),
        )
        prediction, confidence = predictor.predict(feature_vector)
        shap_data = explainer.explain(
            model=predictor._model,
            feature_vector=feature_vector,
            feature_names=feature_names,
            feature_map=feature_map,
            prediction=prediction,
        )

        row = AuditLog(
            batch_id=payload.batch_id,
            prediction=PredictionEnum(prediction.value),
            confidence=float(confidence),
            shap_json=shap_data,
            input_payload=payload.model_dump(mode="json"),
            model_version=predictor.model_version,
            signed_by=settings.signed_by_default,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)

        created_at = row.created_at or datetime.now(timezone.utc)
        return PredictResponse(
            batch_id=payload.batch_id,
            prediction=PredictionLabel(prediction.value),
            confidence=round(float(confidence), 4),
            shap_explanation=shap_data,
            audit={
                "decision_id": row.id,
                "timestamp": created_at,
                "model_version": row.model_version,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction processing failed") from exc


@router.get("/batches", response_model=list[BatchSummaryResponse])
async def list_batches(db: AsyncSession = Depends(get_db_session)) -> list[BatchSummaryResponse]:
    query = select(AuditLog).order_by(AuditLog.created_at.desc())
    result = await db.execute(query)
    rows = result.scalars().all()
    return [
        BatchSummaryResponse(
            decision_id=row.id,
            batch_id=row.batch_id,
            prediction=PredictionLabel(row.prediction.value),
            confidence=row.confidence,
            model_version=row.model_version,
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.get("/batches/{batch_id}", response_model=BatchDetailResponse)
async def get_batch(batch_id: str, db: AsyncSession = Depends(get_db_session)) -> BatchDetailResponse:
    query = select(AuditLog).where(AuditLog.batch_id == batch_id).order_by(AuditLog.created_at.desc())
    result = await db.execute(query)
    row = result.scalars().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    return BatchDetailResponse(
        decision_id=row.id,
        batch_id=row.batch_id,
        prediction=PredictionLabel(row.prediction.value),
        confidence=row.confidence,
        model_version=row.model_version,
        created_at=row.created_at,
        shap_explanation=row.shap_json,
        input_payload=row.input_payload,
        signed_by=row.signed_by,
    )
