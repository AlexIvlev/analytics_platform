import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from settings import Settings
from utils import save_model_meta

settings = Settings()


def fit_model(request):
    X = np.array(request.X)
    y = np.array(request.y)
    model_config = request.config

    model = LogisticRegression()

    hyperparameters = model_config.hyperparameters
    model.set_params(**hyperparameters)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    model_dir = settings.model_storage_dir
    os.makedirs(model_dir, exist_ok=True)
    model_filename = os.path.join(model_dir, f"{model_config.id}.joblib")
    joblib.dump(model, model_filename)

    save_model_meta(
        model_storage_dir=model_dir,
        model_meta_file=settings.model_meta_file,
        model_id=model_config.id,
        description=model_config.description,
        model_type=model_config.type,
        hyperparameters=model_config.hyperparameters,
    )

    return model_config.id
