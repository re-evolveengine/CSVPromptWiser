import streamlit as st

from model.core.llms.prompt_optimizer import PromptOptimizer
from model.io.sqlite_result_saver import SQLiteResultSaver
from streamlit_dir.elements.api_key_ui import load_api_key_ui
from streamlit_dir.elements.dataset_handler_ui import handle_dataset_upload_or_load, configure_and_process_chunks
from streamlit_dir.elements.model_family_selector import llm_selector
from streamlit_dir.elements.model_selector_ui import model_selector_ui
from streamlit_dir.elements.prompt_input_ui import prompt_input_ui
from streamlit_dir.elements.render_export_section import render_export_section
from utils.constants import APP_NAME


def cwp_sidebar():
    st.sidebar.header(f"{APP_NAME} Controls")

    selected_llm = llm_selector(st.sidebar.container())

    if selected_llm is None:
        return None, None, None, None, None, None, None, None, None, None

    # 🔐 Model Configuration Section
    with st.sidebar.expander("🔐 Model Configuration", expanded=False):
        config_container = st.container()
        api_key = load_api_key_ui(config_container)
        
        # Initialize variables
        selected_model = None
        gemini_client = None
        generation_config = None
        prompt_optimizer = None
        total_token_budget = None
        
        # Only try to get model if we have an API key
        if api_key:
            selected_model, gemini_client, generation_config = model_selector_ui(config_container, api_key)
            prompt_optimizer = PromptOptimizer(model_name=selected_model)

    # 📁 Upload Section
    with st.sidebar.expander("📁 Upload Data", expanded=False):
        df, saved_filename = handle_dataset_upload_or_load()

    # ✍️ Prompt Input Section
    with st.sidebar.expander("✍️ Prompt Input", expanded=False):
        prompt_container = st.container()
        prompt, response_example = prompt_input_ui(prompt_container)

    # 🔪 Chunking Section
    with st.sidebar.expander("🔪 Chunk Settings", expanded=False):
        chunk_file_path, chunk_summary = (None, None)
        if df is not None:
            chunk_file_path, chunk_summary, total_token_budget = configure_and_process_chunks(df,prompt, response_example, optimizer=prompt_optimizer)
        else:
            st.info("ℹ️ Upload a file first to configure chunking")

    # 🧩 Configure Chunk Processing Section
    with st.sidebar.expander("🧩 Configure Chunk Processing", expanded=False):
        if not all([gemini_client, prompt, chunk_file_path]):
            st.warning("⚠️ Please complete all previous steps: upload data, enter a prompt, and select a model.")
            # Do NOT reset here unless you explicitly want to cancel readiness
        else:
            with st.form("chunk_processing_form"):
                st.session_state["num_chunks"] = st.number_input(
                    "🔢 Number of chunks to process",
                    min_value=1,
                    max_value=100,
                    value=5,
                    step=1,
                    key="num_chunks_input"
                )

                if st.form_submit_button("⚙️ Set Processing Parameters"):
                    st.session_state["last_status"] = {
                        "processed": 0,
                        "chunk_count": st.session_state["num_chunks"]
                    }

                    st.session_state["processing_ready"] = True

    db_saver = SQLiteResultSaver()

    # The export section will now only appear if a chunk file exists AND the DB is not empty.
    if chunk_file_path and (st.session_state.get("has_results") or db_saver.has_results()):
        with st.sidebar.expander("💾 Export Processed Results", expanded=False):
            render_export_section(chunk_file_path=chunk_file_path)

    return api_key, selected_model, df, chunk_file_path, chunk_summary, prompt,response_example, generation_config, gemini_client, total_token_budget
