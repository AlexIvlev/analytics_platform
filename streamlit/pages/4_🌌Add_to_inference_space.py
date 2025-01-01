import streamlit as st
import logging
import requests
from util.logging_handler import configure_logger

logger = configure_logger(__name__, logging.DEBUG)

st.set_page_config(page_title="Add to inference space", page_icon="🤖")


@st.cache_data
def fetch_inference_space_models():
    response = requests.get(st.session_state.backend_url + "/status")
    if response.status_code != 200:
        logger.error(response.text)
        return []
    logger.debug(response.json())
    return response.json()


st.markdown("# Добавить модель в пространство инференса")
st.sidebar.header("Добавить модель в пространство инференса")
st.write(
    """На этой странице вы можете добавить одну из загруженных ранее моделей в пространство инференса.""")

logger.debug("Inference space page loaded")
