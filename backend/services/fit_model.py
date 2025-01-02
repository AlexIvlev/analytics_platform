import os
import signal
from http import HTTPStatus

import cloudpickle
from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from settings import Settings
from utils import save_model_meta

settings = Settings()


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Training took too long")


def fit_model(request, timeout: int = 10):
    """Выполняет обучение модели с заданным таймаутом"""
    model_config = request.config
    try:
        # Устанавливаем обработчик сигнала
        signal.signal(signal.SIGALRM, timeout_handler)
        # Устанавливаем таймаут
        signal.alarm(timeout)

        X = np.array(request.X)
        y = np.array(request.y)

        model = LogisticRegression()
        hyperparameters = model_config.hyperparameters
        model.set_params(**hyperparameters)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        # Отключаем таймер
        signal.alarm(0)

        # Сохраняем модель
        model_dir = settings.model_storage_dir
        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, f"{model_config.id}.pkl")
        cloudpickle.dump(model, model_filename)

        # Сохраняем метаданные модели
        save_model_meta(
            model_storage_dir=model_dir,
            model_meta_file=settings.model_meta_file,
            model_id=model_config.id,
            description=model_config.description,
            model_type=model_config.type,
            hyperparameters=model_config.hyperparameters,
        )

        return model_config.id

    except TimeoutError:
        print(f"Training timeout for model {model_config.id}")
        raise HTTPException(
            status_code=HTTPStatus.REQUEST_TIMEOUT,
            detail=f"Model {model_config.id} training exceeded timeout limit of 10 seconds"
        )
