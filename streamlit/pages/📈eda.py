import streamlit as st
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

from util.wordcloud import create_wordcloud, create_bigram_cloud
from util.check_dataset import check_uploaded_data

st.set_page_config(page_title="EDA", page_icon="📈")

st.markdown("# EDA")
st.sidebar.header("EDA")
st.write(
    """На этой странице вы можете загрузить датасет и провести его исследовательский анализ (EDA)."""
)

dataset = st.radio("Выберите датасет", ("news :newspaper:", "social🧻"))

st.write(f"Вы выбрали датасет: {dataset}")

uploaded_file = st.file_uploader("Выберите parquet-файл с датасетом", type=["parquet"])

show_wordcloud = st.checkbox("Показать облако слов")


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
