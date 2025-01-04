import streamlit as st
import requests


@st.cache_data
def fetch_models():
    response = requests.get(st.session_state.backend_url + "/models")
    print(response.json())
    return response.json()["models"]
