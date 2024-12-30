import os
import logging
import streamlit as st
from logging.handlers import TimedRotatingFileHandler


LOGS_DIR = "logs"


@st.cache_resource
def configure_logger(module_name, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    logger.addHandler(create_file_handler(LOGS_DIR))
    return logger


def create_file_handler(logs_dir: str) -> TimedRotatingFileHandler:
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, "app.log")

    file_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=7)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    return file_handler
