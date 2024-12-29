import logging

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from util.check_dataset import check_uploaded_data
from util.logging_handler import configure_logger
from util.wordcloud import create_wordcloud, create_bigram_cloud

st.set_page_config(page_title="EDA", page_icon="📈")

logger = configure_logger(__name__, logging.DEBUG)

st.markdown("# EDA")
st.sidebar.header("EDA")
st.write(
    """На этой странице вы можете загрузить датасет и провести его исследовательский анализ (EDA).
    Выберите датасет, загрузите файл и выберите, какие графики вы хотите построить.
    Social и News датасеты имеют разные опции визуализации."""
)

dataset = st.radio("Выберите датасет", ("News :newspaper:", "Social🧻"))

st.write(f"Вы выбрали датасет: {dataset}")

uploaded_file = st.file_uploader("Выберите parquet-файл с датасетом", type=["parquet"])

show_ticker_treemap = st.checkbox("Показать тикеры в виде treemap")
show_wordcloud = st.checkbox("Показать облако слов")
show_subreddit_distribution = st.checkbox("Показать распределение по сабреддитам")
show_relative_price_change = st.checkbox("Показать относительное изменение цен за день")


@st.cache_data
def load_data(file: UploadedFile) -> pd.DataFrame:
    return pd.read_parquet(file)


def plot_wordcloud(df: pd.DataFrame) -> None:
    wordcloud = create_wordcloud(df)
    bigram_cloud = create_bigram_cloud(df)
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

    if show_wordcloud:
        plot_wordcloud(df)

    if show_subreddit_distribution:
        subreddit_counts = df['subreddit'].value_counts().reset_index()
        subreddit_counts.columns = ['subreddit', 'count']

        custom_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

        fig = px.bar(subreddit_counts,
                     x='subreddit',
                     y='count',
                     title='Distribution of Posts by Subreddit',
                     color='subreddit',
                     color_discrete_sequence=custom_colors)
        fig.update_layout(xaxis_title='Subreddit', yaxis_title='Number of Posts')
        st.plotly_chart(fig)

    if show_relative_price_change:
        custom_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

        df['rel_price_change'] = ((df['price_1d'] - df['created_price']) / df['created_price']) * 100
        df = df[df['rel_price_change'] > -100]
        df = df[df['rel_price_change'] < 100]

        fig = px.histogram(
            df,
            x='rel_price_change',
            nbins=200,
            title='Distribution of Price Change %',
            color_discrete_sequence=custom_colors
        )

        fig.update_layout(
            xaxis_title='Relative Price Change',
            yaxis_title='Frequency',
            bargap=0.1,
            width=1000,
            height=500,
        )

        st.plotly_chart(fig)

    if show_ticker_treemap:
        ticker_counts = df['ticker'].value_counts().reset_index()
        ticker_counts.columns = ['ticker', 'count']

        fig = px.treemap(
            ticker_counts,
            path=['ticker'],
            values='count',
            title='Treemap of Ticker Counts'
        )

        st.plotly_chart(fig)

logger.debug("EDA page loaded")