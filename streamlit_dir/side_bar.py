import streamlit as st
from model.utils.constants import APP_NAME
from streamlit_dir.side_bar_utils import model_selector_ui, load_api_key_ui, prompt_input_ui, \
    handle_dataset_upload_or_load
from streamlit_dir.stramlit_dataset_handler import StreamlitDatasetHandler


def cwp_sidebar():
    st.sidebar.header(f"{APP_NAME} Controls")
    
    # Initialize variables
    api_key = None
    selected_model = None
    uploaded_file = None
    df = None
    
    # Model Configuration
    model_config = st.sidebar.expander("🔐 Model Configuration", expanded=True)
    with model_config:
        api_key = load_api_key_ui(st.container())
        if api_key:  # Only show model selection if we have an API key
            selected_model = model_selector_ui(st.container(), api_key)
    
    # Dataset Upload
    upload_section = st.sidebar.expander("📁 Dataset Upload", expanded=True)
    with upload_section:
        df = handle_dataset_upload_or_load()

    # Prompt Input
    prompt_expander = st.sidebar.expander("✍️ Prompt Input", expanded=True)
    with prompt_expander:
        prompt_container = st.container()
        prompt = prompt_input_ui(prompt_container)

    return api_key, selected_model, uploaded_file, df, prompt


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
