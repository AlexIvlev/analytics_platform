import json
import os

import cloudpickle
import pandas as pd

from serializers import ModelType
from settings import Settings
from utils import save_model_meta

settings = Settings()

LEARNING_CURVE_STUB_SOCIAL = [
    1.0, 0.9, 0.81, 0.729, 0.656, 0.590, 0.531, 0.478, 0.430, 0.387,
    0.348, 0.313, 0.282, 0.254, 0.228, 0.205, 0.185, 0.166, 0.150, 0.135,
    0.121, 0.109, 0.098, 0.088, 0.079, 0.071, 0.064, 0.057, 0.051, 0.046,
    0.041, 0.037, 0.033, 0.029, 0.026, 0.024, 0.021, 0.019, 0.017, 0.015,
    0.014, 0.012, 0.011, 0.010, 0.009, 0.008, 0.007, 0.006, 0.006, 0.005
]
LEARNING_CURVE_STUB_NEWS = [
    1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.73, 0.69, 0.66,
    0.63, 0.60, 0.57, 0.54, 0.52, 0.49, 0.47, 0.45, 0.42, 0.40,
    0.38, 0.36, 0.34, 0.33, 0.31, 0.29, 0.28, 0.26, 0.25, 0.24,
    0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13,
    0.12, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07
]


class ModelInferenceManager:
    def __init__(self, settings: Settings, state_file: str = "inference_state.json"):
        self.settings = settings
        self.model_storage_dir = self.settings.model_storage_dir
        self.state_file = os.path.join(self.model_storage_dir, state_file)
        self.loaded_models = {}
        self.loaded_model_ids = {}
        self._load_state()

    def _load_model_into_memory(self, model_id: str):
        """Загрузка модели из файла с использованием cloudpickle.loads"""
        model_path = os.path.join(self.model_storage_dir, f"{model_id}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_id}.pkl' not found at path {model_path}")
        with open(model_path, "rb") as f:
            return cloudpickle.load(f)

    def _save_state(self):
        """Сохранение состояния загрузки моделей (только идентификаторы)"""
        with open(self.state_file, "w") as f:
            json.dump(self.loaded_model_ids, f, ensure_ascii=False, indent=4)

    def _load_state(self):
        """Загрузка состояния из файла JSON"""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                state = json.load(f)
        else:
            state = {
                "social": settings.default_model_social,
                "news": settings.default_model_news
            }

        for model_type, model_id in state.items():
            if model_id:
                self.loaded_models[model_type] = self._load_model_into_memory(model_id)
                self.loaded_model_ids[model_type] = model_id

        with open(self.state_file, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

        save_model_meta(
            model_storage_dir=settings.model_storage_dir,
            model_meta_file=settings.model_meta_file,
            model_id=settings.default_model_social.split('.')[0],
            description="Default social model",
            model_type=ModelType.social,
            hyperparameters={},
            learning_curve=LEARNING_CURVE_STUB_SOCIAL
        )
        save_model_meta(
            model_storage_dir=settings.model_storage_dir,
            model_meta_file=settings.model_meta_file,
            model_id=settings.default_model_news.split('.')[0],
            description="Default news model",
            model_type=ModelType.news,
            hyperparameters={},
            learning_curve=LEARNING_CURVE_STUB_NEWS
        )

    def load(self, model_type: str, model_id: str):
        """
        Загрузка или замена модели для указанного типа. Если модель уже загружена для данного типа - она будет заменена.
        """
        model = self._load_model_into_memory(model_id)

        self.loaded_models[model_type] = model
        self.loaded_model_ids[model_type] = model_id

        self._save_state()

        return f"Model of type '{model_type}' with ID '{model_id}' loaded for inference"

    def get_status(self) -> dict:
        """Получение статуса загруженных моделей"""
        if self.loaded_model_ids:
            return {"status": "loaded", "models": self.loaded_model_ids}
        return {"status": "empty"}

    def predict(self, model_type: str, X: pd.DataFrame) -> list[float]:
        """Предсказание с использованием загруженной модели."""
        if model_type not in self.loaded_models:
            raise ValueError(f"Model of type '{model_type}' is not loaded")
        model = self.loaded_models[model_type]
        return model.predict(X).tolist()
