import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from api.v1.api_route import router as models_router
from serializers import ModelType
from settings import Settings
from multiprocessing import Value, Lock

from utils import save_model_meta

ml_models = {}
settings = Settings()


async def ml_lifespan_manager(app: FastAPI):
    model_path_social = Path(settings.model_storage_dir) / settings.default_model_social
    model_path_news = Path(settings.model_storage_dir) / settings.default_model_news

    # Путь к файлу состояния (если его нет, он будет создан)
    inference_state_path = Path(settings.model_storage_dir) / "inference_state.json"

    try:
        # Если файл состояния не существует или в нем нет нужных ключей, создаем его
        if not inference_state_path.exists():
            state = {
                "social": settings.default_model_social,
                "news": settings.default_model_news
            }

            with open(inference_state_path, "w") as f:
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

        yield
    except Exception:
        pass
    finally:
        ml_models.clear()

app = FastAPI(
    title="backend",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    lifespan=ml_lifespan_manager
)


class StatusResponse(BaseModel):
    status: str
    active_train_processes: int
    max_train_processes: int

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App is healthy"}]}
    )

active_train_processes = Value("i", 0)
lock = Lock()
max_train_processes = settings.num_cores - 1

app.state.active_train_processes = active_train_processes
app.state.lock = lock
app.state.max_train_processes = max_train_processes


@app.get("/")
async def root() -> StatusResponse:
    return StatusResponse(
        status="Server is running",
        active_train_processes=active_train_processes.value,
        max_train_processes=max_train_processes
    )


app.include_router(models_router, prefix="/api/v1")
