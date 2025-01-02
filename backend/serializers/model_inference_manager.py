from typing import Any

from pydantic import BaseModel


class SetRequest(BaseModel):
    model_type: str
    model_id: str


class SetResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    status: str
    models: Any


class PredictRequest(BaseModel):
    id: str
    X: list[list[float]]


class PredictResponse(BaseModel):
    predictions: list[float]
