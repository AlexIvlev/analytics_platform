import streamlit as st
import logging
from util.logging_handler import configure_logger

st.set_page_config(page_title="Models", page_icon="🤖")

logger = configure_logger(__name__, logging.DEBUG)

st.markdown("# Models")
st.sidebar.header("Models")
st.write(
    """На этой странице вы можете посмотреть информацию о существующих моделях"""
)

logger.debug("Models page loaded")
