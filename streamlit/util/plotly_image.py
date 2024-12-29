import io
import plotly.graph_objects as go
from PIL import Image


def create_image(wordcloud, type='Word') -> go.Figure:
    buffer = io.BytesIO()
    wordcloud.to_image().save(buffer, format='PNG')
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
        title=f"{type} Cloud",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig
