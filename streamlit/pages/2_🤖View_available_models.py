import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from util.logging_handler import configure_logger
from util.model_helper import fetch_models

st.set_page_config(page_title="View available models", page_icon="🤖")

logger = configure_logger(__name__, logging.DEBUG)

st.markdown("# Просмотр доступных моделей")
st.sidebar.header("Просмотр доступных моделей")
st.write(
    """На этой странице вы можете посмотреть информацию о существующих моделях и
    сравнить их кривые обучения."""
)


models = fetch_models()

models_df = pd.DataFrame([
    {
        "ID": model["id"],
        "Type": model["type"],
        "Description": model["description"],
        "Hyperparameters": str(model["hyperparameters"]),
        "Has Learning Curve": bool(model["learning_curve"])
    }
    for model in models
])

model_ids = [model["id"] for model in models]

st.subheader("Список моделей")
st.dataframe(models_df)

show_model_details = st.checkbox("Отобразить подробную информацию о модели", value=False)
show_learning_curve_comparison = st.checkbox("Отобразить сравнение кривых обучения", value=False)

if show_model_details:

    selected_model_id = st.selectbox("Выберите модель для просмотра информации:", model_ids)

    selected_model = next(model for model in models if model["id"] == selected_model_id)
    st.write("### Информация о выбранной модели")
    st.json(selected_model)

    if selected_model["learning_curve"]:
        st.write("### Кривая обучения")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=selected_model["learning_curve"],
                mode='lines+markers',
                name=f"Кривая обучения ({selected_model_id})"
            )
        )
        fig.update_layout(
            title=f"Кривая обучения для модели {selected_model_id}",
            xaxis_title="Эпоха",
            yaxis_title="Функция потерь",
            template="plotly_white"
        )
        st.plotly_chart(fig)

    else:
        st.write("Для данной модели нет данных о кривой обучения")

    logger.debug("Models page loaded")

if show_learning_curve_comparison:
    selected_ids = st.multiselect("Выберите модели для сравнения:", model_ids)

    st.write("### Сравнение кривых обучения")
    fig = go.Figure()
    for selected_id in selected_ids:
        model = next(model for model in models if model["id"] == selected_id)
        if model["learning_curve"]:
            fig.add_trace(
                go.Scatter(
                    y=model["learning_curve"],
                    mode='lines+markers',
                    name=f"Кривая обучения ({model['id']})"
                )
            )
    fig.update_layout(
        title="Сравнение кривых обучения",
        xaxis_title="Эпоха",
        yaxis_title="Функция потерь",
        template="plotly_white"
    )
    st.plotly_chart(fig)
