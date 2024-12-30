import io
import plotly.graph_objects as go
from PIL import Image
import plotly.express as px
import pandas as pd
import streamlit as st


@st.cache_data
def plot_ticker_treemap(df: pd.DataFrame) -> go.Figure:
    ticker_counts = df['ticker'].value_counts().reset_index()
    ticker_counts.columns = ['ticker', 'count']

    fig = px.treemap(
        ticker_counts,
        path=['ticker'],
        values='count',
        title='Treemap of Ticker Counts'
    )
    return fig


@st.cache_data
def plot_relative_price_change(df: pd.DataFrame) -> go.Figure:
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
    return fig


@st.cache_data
def plot_subreddits_distribution(df: pd.DataFrame) -> go.Figure:
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
    return fig


@st.cache_data
def create_image(_wordcloud, cloud_type='Word') -> go.Figure:
    buffer = io.BytesIO()
    _wordcloud.to_image().save(buffer, format='PNG')
    buffer.seek(0)
    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=Image.open(buffer),
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            sizex=1,
            sizey=1,
            xanchor="left",
            yanchor="top",
            layer="below"
        )
    )

    fig.update_layout(
        title=f"{cloud_type} Cloud",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig


@st.cache_data
def plot_news_distributions(df: pd.DataFrame) -> (go.Figure, go.Figure, go.Figure, go.Figure):
    article_type_counts = df['articleType'].value_counts().reset_index()
    article_type_counts.columns = ['articleType', 'count']

    fig_atc = px.bar(
        article_type_counts,
        x='count',
        y='articleType',
        orientation='h',
        color='articleType',
        title='Распределение по типам статей',
        labels={'count': 'Количество', 'articleType': 'Тип статьи'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig_atc.update_layout(
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )

    sector_counts = df['Sector'].value_counts().reset_index()
    sector_counts.columns = ['Sector', 'count']

    fig_sc = px.bar(
        sector_counts,
        x='count',
        y='Sector',
        orientation='h',
        color='Sector',
        title='Распределение по секторам',
        labels={'count': 'Количество', 'Sector': 'Сектор'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig_sc.update_layout(
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )

    industry_counts = df['Industry'].value_counts().reset_index()
    industry_counts.columns = ['Industry', 'count']

    fig_ic = px.bar(
        industry_counts,
        x='count',
        y='Industry',
        orientation='h',
        color='Industry',
        title='Распределение по отраслям',
        labels={'count': 'Количество', 'Industry': 'Отрасль'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig_ic.update_layout(
        showlegend=False,
        height=1000,
        yaxis={'categoryorder': 'total ascending'}
    )

    ticker_news_counts = df.groupby('Symbol').size().reset_index(name='news_count')
    ticker_news_counts = ticker_news_counts.sort_values(by='news_count', ascending=False)

    top_tickers = ticker_news_counts.sort_values(by='news_count', ascending=False).head(50)

    fig_tc = px.bar(
        top_tickers,
        x='news_count',
        y='Symbol',
        orientation='h',
        color='Symbol',
        title='Количество новостей для каждого тикера (Top 50)',
        labels={'news_count': 'Количество новостей', 'Symbol': 'Тикер'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig_tc.update_layout(
        showlegend=False,
        height=800
    )

    return fig_atc, fig_sc, fig_ic, fig_tc


@st.cache_data
def plot_temporal_analysis(df: pd.DataFrame) -> (go.Figure, go.Figure, go.Figure, go.Figure):
    df['Date'] = pd.to_datetime(df['Date'])

    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month_name()

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                    'August', 'September', 'October', 'November', 'December']

    df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=days_order, ordered=True)
    df['Month'] = pd.Categorical(df['Month'], categories=months_order, ordered=True)

    day_article_counts = df.groupby(['DayOfWeek', 'articleType'], observed=False).size().unstack(fill_value=0)

    month_article_counts = df.groupby(['Month', 'articleType'], observed=False).size().unstack(fill_value=0)

    day_sector_counts = df.groupby(['DayOfWeek', 'Sector'], observed=False).size().unstack(fill_value=0)

    month_sector_counts = df.groupby(['Month', 'Sector'], observed=False).size().unstack(fill_value=0)

    fig_day_article = px.bar(
        day_article_counts.reset_index().melt(id_vars='DayOfWeek', var_name='articleType', value_name='count'),
        x='DayOfWeek',
        y='count',
        color='articleType',
        title='Распределение типов новостей по дням недели',
        labels={'DayOfWeek': 'День недели', 'count': 'Количество статей', 'articleType': 'Тип статьи'},
        barmode='stack',
        category_orders={'DayOfWeek': days_order},
        color_discrete_sequence=px.colors.sequential.Agsunset
    )
    fig_day_article.update_layout(xaxis_tickangle=45)

    fig_month_article = px.bar(
        month_article_counts.reset_index().melt(id_vars='Month', var_name='articleType', value_name='count'),
        x='Month',
        y='count',
        color='articleType',
        title='Распределение типов новостей по месяцам',
        labels={'Month': 'Месяц', 'count': 'Количество статей', 'articleType': 'Тип статьи'},
        barmode='stack',
        category_orders={'Month': months_order},
        color_discrete_sequence=px.colors.sequential.Agsunset
    )
    fig_month_article.update_layout(xaxis_tickangle=45)

    fig_day_sector = px.bar(
        day_sector_counts.reset_index().melt(id_vars='DayOfWeek', var_name='Sector', value_name='count'),
        x='DayOfWeek',
        y='count',
        color='Sector',
        title='Распределение новостей секторов по дням недели',
        labels={'DayOfWeek': 'День недели', 'count': 'Количество статей', 'Sector': 'Сектор'},
        barmode='stack',
        category_orders={'DayOfWeek': days_order},
        color_discrete_sequence=px.colors.qualitative.D3
    )
    fig_day_sector.update_layout(xaxis_tickangle=45)

    fig_month_sector = px.bar(
        month_sector_counts.reset_index().melt(id_vars='Month', var_name='Sector', value_name='count'),
        x='Month',
        y='count',
        color='Sector',
        title='Распределение новостей секторов по месяцам',
        labels={'Month': 'Месяц', 'count': 'Количество статей', 'Sector': 'Сектор'},
        barmode='stack',
        category_orders={'Month': months_order},
        color_discrete_sequence=px.colors.qualitative.D3
    )
    fig_month_sector.update_layout(xaxis_tickangle=45)

    return fig_day_article, fig_month_article, fig_day_sector, fig_month_sector


