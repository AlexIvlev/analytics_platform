import json
import os

import cloudpickle

from serializers import ModelType
from settings import Settings
from utils import save_model_meta

settings = Settings()


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
        )
        save_model_meta(
            model_storage_dir=settings.model_storage_dir,
            model_meta_file=settings.model_meta_file,
            model_id=settings.default_model_news.split('.')[0],
            description="Default news model",
            model_type=ModelType.news,
            hyperparameters={},
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

    def predict(self, model_type: str, X: list[list[float]]) -> list[float]:
        """Предсказание с использованием загруженной модели"""
        if model_type not in self.loaded_models:
            raise ValueError(f"Model of type '{model_type}' is not loaded")
        model = self.loaded_models[model_type]
        return model.predict(X).tolist()
