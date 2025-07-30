import streamlit as st
from model.utils.constants import APP_NAME
from streamlit_dir.ui.api_key_ui import load_api_key_ui
from streamlit_dir.ui.dataset_handler_ui import handle_dataset_upload_or_load_and_chunk
from streamlit_dir.ui.model_selector_ui import model_selector_ui
from streamlit_dir.ui.prompt_input_ui import prompt_input_ui


def cwp_sidebar():
    st.sidebar.header(f"{APP_NAME} Controls")

    # 🔐 Model Configuration Section
    with st.sidebar.expander("🔐 Model Configuration", expanded=True) as model_config:
        with st.sidebar.expander("🔐 Model Configuration", expanded=True):
            config_container = st.container()
            api_key = load_api_key_ui(config_container)
            selected_model = model_selector_ui(config_container, api_key) if api_key else None

    # 📁 Upload & Chunk Section
    with st.sidebar.expander("📁 Upload & Chunk", expanded=True) as upload_chunk:
        df, saved_filename, chunk_summary = handle_dataset_upload_or_load_and_chunk()

    # ✍️ Prompt Input Section
    with st.sidebar.expander("✍️ Prompt Input", expanded=True):
        prompt_container = st.container()
        prompt = prompt_input_ui(prompt_container)

    return api_key, selected_model, df, saved_filename, chunk_summary, prompt


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
    # return api_key, selected_model, uploaded_file
