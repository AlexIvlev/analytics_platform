from enum import Enum

from pydantic import BaseModel
from typing import Any


class ModelType(str, Enum):
    social = "social"
    news = "news"


class ModelConfig(BaseModel):
    id: str
    description: str
    type: str
    hyperparameters: dict[str, Any]


class FitRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    config: ModelConfig


class FitResponse(BaseModel):
    message: str
