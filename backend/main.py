import sys
import os

sys.path.append(os.path.abspath("../ml"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict

app = FastAPI(title="AnomAlert Bio API")


class BatchInput(BaseModel):
    pH_mean: float
    temp_max: float
    oxygen_variance: float
    ts_mean: float
    ts_std: float
    ts_max: float
    ts_min: float


@app.get("/")
def root():
    return {"message": "API working"}


@app.post("/predict")
def predict_batch(input: BatchInput):
    try:
        result = predict(input.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))