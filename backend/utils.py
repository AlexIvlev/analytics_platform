import os
import json


def save_model_meta(model_storage_dir, model_meta_file, model_id, description, model_type, hyperparameters):
    """
    Сохраняет метаинформацию о модели в файл json.

    :param model_storage_dir: Папка для хранения моделей.
    :param model_meta_file: Имя файла для метаинформации.
    :param model_id: ID модели.
    :param description: Описание модели.
    :param model_type: Тип модели.
    :param hyperparameters: Гиперпараметры модели.
    """
    meta_path = os.path.join(model_storage_dir, model_meta_file)

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            meta_data = json.load(meta_file)
    else:
        meta_data = {}

    meta_data[model_id] = {
        "description": description,
        "type": model_type,
        "hyperparameters": hyperparameters,
    }

    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(meta_data, meta_file, indent=4, ensure_ascii=False)
