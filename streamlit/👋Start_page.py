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
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

st.session_state["backend_url"] = BACKEND_URL

logger.debug("Main page loaded")
