import os
import signal
from http import HTTPStatus

import cloudpickle
import pandas as pd
from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipelines.social_pipeline import CustomTransformer, preprocessor
from serializers.fit_model import ModelConfig
from settings import Settings
from utils import save_model_meta

settings = Settings()


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Training took too long")


def fit_model(dataframe: pd.DataFrame, config: ModelConfig, timeout: int = 10):
    """Выполняет обучение модели с заданным таймаутом"""
    model_config = config
    try:
        # Устанавливаем обработчик сигнала
        signal.signal(signal.SIGALRM, timeout_handler)
        # Устанавливаем таймаут
        signal.alarm(timeout)

        dataframe['target'] = np.where(
            dataframe['price_1d'] > dataframe['created_price'], 1, 0
        )
        X = dataframe.drop(columns=['target'])
        y = dataframe['target']

        social_pipeline = Pipeline([
            ('custom', CustomTransformer()),
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**config.hyperparameters))
        ])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        social_pipeline.fit(X_train, y_train)

        # Отключаем таймер
        signal.alarm(0)

        # Сохраняем модель
        model_dir = settings.model_storage_dir
        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, f"{model_config.id}.pkl")
        with open(model_filename, 'wb') as file:
            cloudpickle.dump(social_pipeline, file)

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
