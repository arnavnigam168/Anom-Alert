import sys
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Robust path wiring from backend -> ml
BASE_DIR = os.path.dirname(__file__)
ML_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "ml"))
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)

from predict import predict

app = FastAPI(title="AnomAlert Bio API")

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


@app.get("/")
def root():
    return {"message": "API working"}


@app.post("/predict")
def predict_batch(input: BatchInput):
    try:
        payload = input.model_dump()
        print("[Backend][/predict] input received:", payload)
        result = predict(payload)
        print("[Backend][/predict] result:", result)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )