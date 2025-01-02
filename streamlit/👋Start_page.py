import logging
import os

import streamlit as st

from util.logging_handler import configure_logger

BACKEND_URL = os.getenv("BACKEND_URL")

st.set_page_config(
    page_title="Start page",
    page_icon="👋",
)
logger = configure_logger(__name__, logging.INFO)

st.write("# Выберите один из предложенных вариантов! 👋")
st.sidebar.success("Выберите одну из опций выше")

st.markdown(
    """
### Аналитическая платформа для предсказания направления движения цен на акции
Представленная аналитическая платформа позволяет сделать предсказание направления движения цен на акции.
Она позволяет:
- Производить анализ данных по загруженному датасету
- Просматривать доступные модели
- Сравнивать кривые обучения
- Обучать новые модели
- Добавлять модели в пространство инференса
- Делать предсказания с помощью моделей, добавленных в пространство инференса

Выберите одну из опций в меню слева для продолжения работы.
    """
)

st.session_state["backend_url"] = BACKEND_URL

logger.debug("Main page loaded")
