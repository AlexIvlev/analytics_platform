import io
import logging

import pandas as pd
import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from util.check_dataset import check_uploaded_data
from util.logging_handler import configure_logger

st.set_page_config(page_title="Predict", page_icon="🔮")

logger = configure_logger(__name__, logging.DEBUG)


@st.cache_data
def load_data(file: UploadedFile) -> pd.DataFrame:
    return pd.read_parquet(file)


@st.cache_data
def fetch_inference_space_models():
    response = requests.get(st.session_state.backend_url + "/status")
    if response.status_code != 200:
        logger.error(response.text)
        return []
    logger.debug(response.json())
    return response.json()


@st.cache_data
def predict(model_id, model_type, data):
    parquet_buffer = io.BytesIO()
    data.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)

    files = {
        "file": ("data.parquet", parquet_buffer, "application/octet-stream")
    }
    payload = {"id": model_id, "model_type": model_type}

    response = requests.post(
        st.session_state.backend_url + "/predict",
        data=payload,
        files=files
    )

    if response.status_code != 200:
        logger.error(response.text)
        return []

    logger.debug(response.json())
    return response.json()


st.markdown("# Сделать предсказание")
st.sidebar.header("Сделать предсказание")
st.write(
    """На этой странице вы можете сделать предсказазание с помощью помещённой в пространство инференса модели.
    Для добавление модели в пространство инференса выберите соответствующий пункт меню.""")

model_type = st.radio("Выберите тип модели", ("Social 🧻", "News 📰"))
model_type_normalized = "social" if model_type == "Social 🧻" else "news"

st.write(f"Вы выбрали тип модели: {model_type}")

uploaded_file = st.file_uploader("Выберите parquet-файл с датасетом для предсказания", type=["parquet"])

if uploaded_file:
    df = load_data(uploaded_file)
    correct, error_message = check_uploaded_data(df, model_type)
    if correct:
        st.write("Данные успешно провалидированы!")
    else:
        st.error(error_message)

    models = fetch_inference_space_models()['models']

    model_id = st.selectbox("Выберите модель", list(models.keys()))

    predict_button = st.button("Сделать предсказание")

    if predict_button:
        st.write("Предсказание:")
        prediction = predict(model_id, model_type_normalized, df)

        st.json(prediction)
