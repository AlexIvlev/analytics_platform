import io
import logging
import uuid

import pandas as pd
import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from util.check_dataset import check_uploaded_data
from util.logging_handler import configure_logger

st.set_page_config(page_title="Fit", page_icon="🎓")

logger = configure_logger(__name__, logging.DEBUG)

st.markdown("# Обучить новую модель")
st.sidebar.header("Обучить новую модель")
st.write(
    """На этой странице вы отправить модель на обучение.
    Вам будет необходимо указать датасет и гиперпараметры модели.""")


@st.cache_data
def load_data(file: UploadedFile) -> pd.DataFrame:
    return pd.read_parquet(file)


@st.cache_data
def fit_model(data: pd.DataFrame, model_config: dict) -> None:
    parquet_buffer = io.BytesIO()
    data.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)

    files = {
        "file": ("data.parquet", parquet_buffer, "application/octet-stream")
    }

    response = requests.post(
        st.session_state.backend_url + "/fit",
        data=model_config,
        files=files
    )

    if response.status_code != 200:
        logger.error(response.text)
        return []
    logger.debug(response.json())
    return response.json()


model_type = st.radio("Выберите тип модели", ("Social 🧻", "News 📰"))
model_type_normalized = "social" if model_type == "Social 🧻" else "news"
st.write(f"Вы выбрали тип модели: {model_type}")

uploaded_file = st.file_uploader("Выберите parquet-файл с датасетом для обучения", type=["parquet"])

if uploaded_file:
    df = load_data(uploaded_file)
    correct, error_message = check_uploaded_data(df, model_type, True)
    if correct:
        st.write("Данные успешно провалидированы!")
    else:
        st.error(error_message)

    desc = st.text_input("Введите описание модели", value="Моя первая модель")
    hyper = st.text_input("Введите гиперпараметры модели в формате JSON", value='{"n_estimators": 100, "max_depth": 5}')
    model_id = str(uuid.uuid4())

    model_config = {
        "id": model_id,
        "type": model_type_normalized,
        "description": desc,
        "hyperparameters": hyper
    }

    fit_model_button = st.button("Обучить модель")
    if fit_model_button:

        response = requests.post(st.session_state.backend_url + "/fit", json=model_config)
        if response.status_code != 200:
            logger.error(response.text)
            st.error("Произошла ошибка при обучении модели")
        else:
            logger.debug(response.json())
            st.success("Процесс обучения запущен!")
