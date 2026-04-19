import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("anomalert.api")

# Robust path wiring from backend -> ml
BASE_DIR = os.path.dirname(__file__)
ML_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "ml"))
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)

from predict import predict, warm_artifacts  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        warm_artifacts()
        logger.info("ML model and scaler ready at startup.")
    except Exception:
        logger.exception("Startup ML warm failed; /predict will attempt lazy load or return ERROR.")
    yield


app = FastAPI(title="AnomAlert Bio API", lifespan=lifespan)

# Temporary dev-friendly CORS for frontend integration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatchInput(BaseModel):
    pH_mean: float
    temp_max: float
    oxygen_variance: float
    ts_mean: float
    ts_std: float
    ts_max: float
    ts_min: float
    ts_slope: float = 0.0
    ts_max_jump: float = 0.0
    ts_rolling_var_mean: float = 0.0
    ts_spike_count: float = 0.0


def _prediction_error_response(message: str) -> dict:
    return {
        "prediction": "ERROR",
        "confidence": 0.0,
        "top_features": [],
        "explanation": message,
    }


@app.get("/")
def root():
    return {"message": "API working"}


@app.post("/predict")
def predict_batch(input: BatchInput):
    """
    Runs batch QC inference. Never raises for model failures: returns structured ERROR payload.
    """
    try:
        payload = input.model_dump()
        logger.info("[Backend][/predict] incoming payload=%s", payload)
        result = predict(payload)
        logger.info("[Backend][/predict] result=%s", result)
        if not isinstance(result, dict):
            logger.error("predict() returned non-dict: %s", type(result))
            return _prediction_error_response("Invalid response from prediction service.")
        return result
    except Exception as exc:
        logger.exception("[Backend][/predict] unexpected failure: %s", exc)
        return _prediction_error_response(f"Prediction request failed: {exc}")
