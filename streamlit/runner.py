import logging

import streamlit as st
from streamlit.web.bootstrap import run

from util.logging_handler import create_file_handler

st.write("# Выберите один из предложенных вариантов! 👋")

for logger_name in logging.root.manager.loggerDict:
    if logger_name.startswith("streamlit"):
        streamlit_logger = logging.getLogger(logger_name)
        streamlit_logger.addHandler(create_file_handler())


main_module = '👋Start_page.py'
run(main_module, False, [], {})
