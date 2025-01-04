import logging

import requests
import streamlit as st

from util.logging_handler import configure_logger
from util.model_helper import fetch_models


@st.cache_data
def fetch_inference_space_models():
    response = requests.get(st.session_state.backend_url + "/status")
    if response.status_code != 200:
        logger.error(response.text)
        return []
    logger.debug(response.json())
    return response.json()


@st.cache_data
def add_to_inference_space(model_type, model_id):
    response = requests.post(st.session_state.backend_url + "/set",
                             json={"model_type": model_type, "model_id": model_id})
    if response.status_code != 200:
        logger.error(response.text)
        return []
    logger.debug(response.json())
    return response.json()


st.set_page_config(page_title="Add to inference space", page_icon="🤖")
logger = configure_logger(__name__, logging.DEBUG)


logger.debug("Inference space page loaded")

st.markdown("# Добавить модель в пространство инференса")
st.sidebar.header("Добавить модель в пространство инференса")
st.write(
    """На этой странице вы можете добавить одну из загруженных ранее моделей в пространство инференса.""")

st.write("Текущее состояние пространства инференса:")
inference_space_models = fetch_inference_space_models()
st.json(inference_space_models)

trained_models = fetch_models()
model_ids = [model["id"] for model in trained_models]
selected_model_id = st.selectbox("Выберите модель для добавления в пространство инференса:", model_ids)
if selected_model_id:
    selected_model_type = next(model["type"] for model in trained_models if model["id"] == selected_model_id)
    add_to_inference_space_button = st.button("Добавить модель в пространство инференса")
    if add_to_inference_space_button:
        add_to_inference_space(selected_model_type, selected_model_id)
