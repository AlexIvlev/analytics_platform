import json
import os
from settings import Settings
from serializers import ModelListItem

settings = Settings()


class ModelStorageManager:
    def __init__(self, model_storage_dir, model_meta_file):
        self.model_storage_dir = model_storage_dir
        self.model_meta_file = model_meta_file

    def list_models(self) -> list[ModelListItem]:
        try:
            meta_path = os.path.join(self.model_storage_dir, self.model_meta_file)

            if not os.path.exists(meta_path):
                return []

            with open(meta_path, "r", encoding="utf-8") as meta_file:
                meta_data = json.load(meta_file)

            models_list = []
            for model_id, model_info in meta_data.items():
                model_list_item = ModelListItem(
                    id=model_id,
                    type=model_info["type"],
                    description=model_info["description"],
                    hyperparameters=model_info["hyperparameters"],
                    learning_curve=model_info["learning_curve"]
                )
                models_list.append(model_list_item)

            return models_list

        except Exception as e:
            raise Exception(f"Error listing models: {str(e)}")

    def remove_model(self, model_id: str):
        try:
            default_model_social = settings.default_model_social
            default_model_news= settings.default_model_news

            if model_id in (default_model_social, default_model_news):
                # Если модель дефолтная, пропускаем удаление
                return False

            model_path = os.path.join(self.model_storage_dir, model_id + ".joblib")
            if os.path.exists(model_path):
                os.remove(model_path)
            else:
                return False

            meta_path = os.path.join(self.model_storage_dir, settings.model_meta_file)
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as meta_file:
                    meta_data = json.load(meta_file)

                if model_id in meta_data:
                    del meta_data[model_id]
                    with open(meta_path, "w", encoding="utf-8") as meta_file:
                        json.dump(meta_data, meta_file, indent=4, ensure_ascii=False)
            return True

        except Exception as e:
            raise Exception(f"Error removing model: {str(e)}")
