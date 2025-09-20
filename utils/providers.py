import streamlit as st
from model.io.model_prefs import ModelPreference

@st.cache_resource
def get_model_prefs():
    return ModelPreference()