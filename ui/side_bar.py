import streamlit as st
from pathlib import Path
from model.utils.cli_utils import load_api_key, handle_model_selection
from model.utils.constants import APP_NAME
from ui.utils.side_bar_utils import model_selector_ui, load_api_key_ui


def sidebar():
    st.sidebar.header(f"{APP_NAME} Controls")

    # Step 0: Load API Key and Fetch Available Models
    api_key = load_api_key_ui()

    selected_model = model_selector_ui(api_key)

    # # File upload (CSV/Parquet)
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV or Parquet file", type=["csv", "parquet"])
    #
    # # Prompt input
    # prompt = st.sidebar.text_area("💬 Enter your prompt")
    #
    # # Chunk size
    # chunk_size = st.sidebar.number_input("🔢 Chunk size", min_value=1, value=100)
    #
    # # Number of chunks to process
    # num_chunks = st.sidebar.number_input("⚙️ Number of chunks to process", min_value=1, value=5)
    #
    # # Start button
    # start_processing = st.sidebar.button("🚀 Start Processing")

    # return uploaded_file, prompt, model_name, api_key, chunk_size, num_chunks, start_processing
    return api_key, selected_model, uploaded_file

