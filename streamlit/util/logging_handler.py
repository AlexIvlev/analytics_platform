import logging
import os
from logging.handlers import TimedRotatingFileHandler

import streamlit as st

LOGS_DIR = os.getenv("LOGS_DIR")


@st.cache_resource
def configure_logger(module_name, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    logger.addHandler(create_file_handler())
    return logger


def create_file_handler() -> TimedRotatingFileHandler:
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(LOGS_DIR, "app.log")

    file_handler = TimedRotatingFileHandler(log_file_path, when="M", interval=10, backupCount=7)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    return file_handler
