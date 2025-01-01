import logging

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from util.check_dataset import check_uploaded_data
from util.logging_handler import configure_logger
from util.plotly_helpers import (plot_relative_price_change,
                                 plot_subreddits_distribution,
                                 plot_temporal_analysis,
                                 plot_ticker_treemap,
                                 plot_news_distributions)
from util.wordcloud import create_wordcloud, create_bigram_cloud

st.set_page_config(page_title="Exploratory data analysis", page_icon="📈")

logger = configure_logger(__name__, logging.DEBUG)

st.markdown("# Разведочный анализ данных")
st.sidebar.header("Разведочный анализ данных")
st.write(
    """На этой странице вы можете загрузить датасет и провести его исследовательский анализ (EDA).
    Выберите датасет, загрузите файл и выберите, какие графики вы хотите построить.
    Social и News датасеты имеют разные опции визуализации."""
)

dataset = st.radio("Выберите датасет", ("News 📰", "Social 🧻"))

st.write(f"Вы выбрали датасет: {dataset}")

uploaded_file = st.file_uploader("Выберите parquet-файл с датасетом", type=["parquet"])


@st.cache_data
def load_data(file: UploadedFile) -> pd.DataFrame:
    return pd.read_parquet(file)


@st.cache_data
def plot_wordcloud(data: pd.DataFrame) -> None:
    wordcloud = create_wordcloud(data)
    bigram_cloud = create_bigram_cloud(data)
    st.plotly_chart(wordcloud, key='wordcloud')
    st.plotly_chart(bigram_cloud, key='bigram_wordcloud')


if uploaded_file:
    df = load_data(uploaded_file)
    correct, error_message = check_uploaded_data(df, dataset)
    if correct:
        st.write("Данные успешно провалидированы!")
    else:
        st.error(error_message)
    st.dataframe(df)

    if dataset == "Social 🧻" and correct:
        show_subreddit_distribution = st.checkbox("Показать распределение по сабреддитам")
        show_ticker_treemap = st.checkbox("Показать тикеры в виде treemap")
        show_relative_price_change = st.checkbox("Показать относительное изменение цен за день")
        show_wordcloud = st.checkbox("Показать облако слов")

        if show_subreddit_distribution:
            st.plotly_chart(plot_subreddits_distribution(df))

        if show_ticker_treemap:
            st.plotly_chart(plot_ticker_treemap(df))

        if show_relative_price_change:
            st.plotly_chart(plot_relative_price_change(df))

        if show_wordcloud:
            plot_wordcloud(df)
    elif dataset == "News 📰" and correct:
        show_distributions = st.checkbox("Показать распределения")
        show_temporal_analysis = st.checkbox("Показать временную динамику статей по типам и секторам")

        if show_distributions:
            figures = plot_news_distributions(df)
            for fig in figures:
                st.plotly_chart(fig)

        if show_temporal_analysis:
            figures = plot_temporal_analysis(df)
            for fig in figures:
                st.plotly_chart(fig)


logger.debug("EDA page loaded")
