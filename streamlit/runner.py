import logging

import streamlit as st
from streamlit.web.bootstrap import run

from util.logging_handler import create_file_handler

LOGS_DIR = "logs"

st.write("# Выберите один из предложенных вариантов! 👋")

for logger_name in logging.root.manager.loggerDict:
    if logger_name.startswith("streamlit"):
        streamlit_logger = logging.getLogger(logger_name)
        streamlit_logger.setLevel(logging.DEBUG)
        streamlit_logger.addHandler(create_file_handler(LOGS_DIR))


main_module = '👋app.py'
run(main_module, False, [], {})
