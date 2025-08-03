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
    with st.sidebar.expander("🔐 Model Configuration", expanded=False):
        config_container = st.container()
        api_key = load_api_key_ui(config_container)
        
        # Initialize variables
        selected_model = None
        gemini_client = None
        generation_config = None
        
        # Only try to get model if we have an API key
        if api_key:
            selected_model, gemini_client, generation_config = model_selector_ui(config_container, api_key)

    # 📁 Upload & Chunk Section
    with st.sidebar.expander("📁 Upload & Chunk", expanded=False):
        df, saved_filename, chunk_file_path, chunk_summary = handle_dataset_upload_or_load_and_chunk(client=gemini_client)

    # ✍️ Prompt Input Section
    with st.sidebar.expander("✍️ Prompt Input", expanded=False):
        prompt_container = st.container()
        prompt = prompt_input_ui(prompt_container)

    with st.sidebar.expander("🧩 Process Chunks", expanded=False):
        if not all([gemini_client, prompt, chunk_file_path]):
            st.warning("⚠️ Please complete all previous steps: upload data, enter a prompt, and select a model.")
            st.session_state["start_processing"] = False
        else:
            with st.form("chunk_processing_form"):
                st.session_state["max_tokens_to_spend"] = st.number_input(
                    "💰 Max tokens to spend",
                    min_value=1000,
                    max_value=100000,
                    value=10000,
                    step=1000,
                    help="Maximum number of tokens to use for this process"
                )
                st.session_state["num_chunks"] = st.number_input(
                    "🔢 Number of chunks to process",
                    min_value=1,
                    max_value=100,
                    value=5,
                    step=1,
                    key="num_chunks_input"
                )
                if st.form_submit_button("🚀 Start Processing"):
                    st.session_state["start_processing"] = True
                else:
                    # Only reset if the form wasn't submitted
                    if "start_processing" not in st.session_state:
                        st.session_state["start_processing"] = False

    return api_key, selected_model, df, chunk_file_path, chunk_summary, prompt, generation_config, gemini_client
