import streamlit as st
from model.utils.constants import APP_NAME
from streamlit_dir.ui.api_key_ui import load_api_key_ui
from streamlit_dir.ui.chunk_processor_panel import process_chunks_ui
from streamlit_dir.ui.dataset_handler_ui import handle_dataset_upload_or_load_and_chunk
from streamlit_dir.ui.model_selector_ui import model_selector_ui
from streamlit_dir.ui.prompt_input_ui import prompt_input_ui


def cwp_sidebar():
    st.sidebar.header(f"{APP_NAME} Controls")

    # 🔐 Model Configuration Section
    with st.sidebar.expander("🔐 Model Configuration", expanded=True):
        config_container = st.container()
        api_key = load_api_key_ui(config_container)
        
        # Initialize variables
        selected_model = None
        gemini_client = None
        
        # Only try to get model if we have an API key
        if api_key:
            selected_model, gemini_client = model_selector_ui(config_container, api_key)

    # 📁 Upload & Chunk Section
    with st.sidebar.expander("📁 Upload & Chunk", expanded=True):
        df, saved_filename, chunk_file_path, chunk_summary = handle_dataset_upload_or_load_and_chunk()

    # ✍️ Prompt Input Section
    with st.sidebar.expander("✍️ Prompt Input", expanded=True):
        prompt_container = st.container()
        prompt = prompt_input_ui(prompt_container)

    # 🧩 Chunk Processor Panel (main body)
    with st.expander("🧩 Process Chunks", expanded=True):
        if not all([gemini_client, prompt, chunk_file_path]):
            st.warning("⚠️ Please complete all previous steps: upload data, enter a prompt, and select a model.")
        else:
            process_chunks_ui(gemini_client, prompt, chunk_file_path)



    return api_key, selected_model, df, saved_filename, chunk_summary, prompt



    # # Number of chunks to process
    # num_chunks = st.sidebar.number_input("⚙️ Number of chunks to process", min_value=1, value=5)
    #
    # # Start button
    # start_processing = st.sidebar.button("🚀 Start Processing")

    # return uploaded_file, prompt, model_name, api_key, chunk_size, num_chunks, start_processing
    # return api_key, selected_model, uploaded_file
