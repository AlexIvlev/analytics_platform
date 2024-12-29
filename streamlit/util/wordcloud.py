from collections import Counter

import plotly.graph_objects as go
from pandas import DataFrame
from wordcloud import WordCloud

from .plotly_image import create_image
from nltk import ngrams


def create_wordcloud(df: DataFrame) -> go.Figure:
    all_text = df['processed_text'].str.cat(sep=' ').lower()
    all_words = all_text.split()

    word_counts = Counter(all_words)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

    return create_image(wordcloud)


def create_bigram_cloud(df: DataFrame) -> go.Figure:
    all_text = df['processed_text'].str.cat(sep=' ').lower()
    all_words = all_text.split()

    bigrams = list(ngrams(all_words, 2))
    bigram_counts = Counter(bigrams)
    bigram_dict = {" ".join(bigram): count for bigram, count in bigram_counts.items()}
    bigram_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigram_dict)

    return create_image(bigram_wordcloud, type='Bigram')
