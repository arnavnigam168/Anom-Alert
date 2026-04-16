import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.database import Base


class PredictionEnum(str, enum.Enum):
    PASS = "PASS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAIL = "FAIL"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    prediction: Mapped[PredictionEnum] = mapped_column(
        Enum(PredictionEnum, name="prediction_enum"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    shap_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    input_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    signed_by: Mapped[str] = mapped_column(String(255), nullable=False)
