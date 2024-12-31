from typing import Any

from pydantic import BaseModel

from serializers import ModelType


class ModelResponse(BaseModel):
    message: str


class ModelListItem(BaseModel):
    id: str
    description: str
    type: ModelType
    hyperparameters: dict[str, Any]
    learning_curve: list[float]


class ModelListResponse(BaseModel):
    models: list[ModelListItem]
