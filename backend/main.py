from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict
from api.v1.api_route import router as models_router
from settings import Settings
from multiprocessing import Value, Lock

from utils import configure_logger

settings = Settings()
app_logger = configure_logger("fastapi_backend")

app = FastAPI(
    title="backend",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",

)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логирование запросов и ответов"""
    app_logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        app_logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        app_logger.error(f"Request failed: {e}")
        raise


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
