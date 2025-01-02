from multiprocessing import Process
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, Path, Body
from http import HTTPStatus

from serializers import FitRequest, FitResponse, ModelResponse, PredictResponse, PredictRequest, \
    SetResponse, SetRequest, StatusResponse, ModelListResponse
from services import fit_model, ModelStorageManager, ModelInferenceManager
from settings import Settings

router = APIRouter()
settings = Settings()
model_storage = ModelStorageManager(
    model_storage_dir=settings.model_storage_dir,
    model_meta_file=settings.model_meta_file
)
model_manager = ModelInferenceManager(
    settings=settings
)


@router.post("/fit", status_code=HTTPStatus.CREATED, response_model=FitResponse)
async def fit(
    request: Annotated[FitRequest, Body(description="Запрос на запуск обучения модели")],
    req: Request
):
    """Запуск обучения модели в отдельном процессе."""
    active_train_processes = req.app.state.active_train_processes
    lock = req.app.state.lock
    max_train_processes = req.app.state.max_train_processes

    with lock:
        if active_train_processes.value >= max_train_processes:
            raise HTTPException(
                status_code=HTTPStatus.TOO_MANY_REQUESTS,
                detail=f"Failed to train model {request.config.id}: No available processes"
            )
        active_train_processes.value += 1

    try:
        process = Process(target=fit_model, args=(request,), kwargs={"timeout": 10})
        process.start()
        process.join()

        if process.exitcode != 0:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f"Model {request.config.id} training failed"
            )

        return FitResponse(message=f"Model {request.config.id} training completed successfully")

    finally:
        with lock:
            active_train_processes.value -= 1


@router.post("/set", response_model=SetResponse)
async def set(request: Annotated[SetRequest, Body(description="Запрос на установку активной модели")]):
    """Установка активных моделей."""
    try:
        message = model_manager.load(request.model_type, request.model_id)
        return SetResponse(message=message)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))


@router.get("/status", response_model=StatusResponse)
async def status():
    """Получение информации о загруженных моделях."""
    status = model_manager.get_status()
    return StatusResponse(status=status['status'], models=status.get('models'))


@router.post("/predict", response_model=PredictResponse)
async def predict(
    model_type: Annotated[str, Form()],
    file: Annotated[UploadFile, File(description="Parquet-файл с данными для предсказания")]
):
    """Предсказание от загруженной модели."""
    try:
        predictions = model_manager.predict(request.id, request.X)
        return PredictResponse(predictions=predictions)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.get("/models", response_model=ModelListResponse)
async def models():
    """Список текущих моделей на сервере с подробной информацией о них."""
    try:
        models = model_storage.list_models()
        return ModelListResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/remove/{model_id}", response_model=list[ModelResponse])
async def remove(model_id: Annotated[str, Path(description="ID удаляемой модели")]):
    """Удаление обученной модели (кроме дефолтных)."""
    try:
        success = model_storage.remove_model(model_id)
        if success:
            return [ModelResponse(message=f"Model '{model_id}' removed")]
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/remove_all", response_model=list[ModelResponse])
async def remove_all():
    """Удаление всех обученных моделей (кроме дефолтных)."""
    try:
        models_list = model_storage.list_models()

        if not models_list:
            return [ModelResponse(message="No models to remove")]

        response = []
        for model in models_list:
            model_id = model.id
            success = model_storage.remove_model(model_id)
            if success:
                response.append(ModelResponse(message=f"Model '{model_id}' removed"))

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
