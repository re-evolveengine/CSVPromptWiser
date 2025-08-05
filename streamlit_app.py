import sys
import os
import streamlit as st
import plotly.graph_objects as go

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from model.utils.constants import APP_NAME, STREAMLIT_CSS_STYLES
from streamlit_dir.side_bar import cwp_sidebar
from streamlit_dir.ui.chunk_processor_panel import process_chunks_ui
from streamlit_dir.ui.token_usage_gauge import render_token_usage_gauge

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    initial_sidebar_state="expanded",
    layout="wide"
)

st.markdown(STREAMLIT_CSS_STYLES,unsafe_allow_html=True)


def main():
    st.title(f"🤖 {APP_NAME} Dashboard")

    # --- Sidebar interaction ---
    api_key, model_name, df, chunk_file_path, chunk_summary, prompt, response_example, generation_config, gemini_client, total_tokens = cwp_sidebar()

    # Initialize session state for expanded state
    if 'expanders' not in st.session_state:
        st.session_state.expanders = {
            'model_config': True,
            'chunk_summary': True,
            'dataset_preview': True,
            'prompt': True
        }

    # Update expanded state if processing starts
    if "start_processing" in st.session_state and st.session_state.get("start_processing"):
        st.session_state.expanders = {k: False for k in st.session_state.expanders}

    # --- Model Configuration ---
    if model_name and generation_config:
        with st.expander("⚙️ Model Configuration", expanded=st.session_state.expanders.get('model_config', True)):
            st.metric("Model", model_name)
            st.metric("Temperature", f"{generation_config.get('temperature', 0.2):.2f}")
            st.metric("Top K", generation_config.get('top_k', 40))
            st.metric("Top P", f"{generation_config.get('top_p', 1.0):.2f}")

    # --- Chunk Summary ---
    if chunk_summary:
        with st.expander("📦 Chunk Summary", expanded=st.session_state.expanders.get('chunk_summary', True)):
            for key, value in chunk_summary.items():
                st.markdown(f"- **{key.replace('_', ' ').capitalize()}**: {value}")

    # --- Dataset Preview ---
    if df is not None:
        with st.expander("📊 Dataset Preview", expanded=st.session_state.expanders.get('dataset_preview', True)):
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(5), use_container_width=True)

    # --- Prompt Echo ---
    if prompt:
        with st.expander("💬 Your Prompt", expanded=st.session_state.expanders.get('prompt', True)):
            st.code(prompt, language="markdown")

    # --- Chunk Processing Output (Always visible) ---
    if gemini_client and prompt and chunk_file_path:
        process_chunks_ui(
            gemini_client,
            prompt,
            chunk_file_path,
            chunk_count=st.session_state.get("num_chunks", 5),
            total_tokens=total_tokens
        )
    else:
        st.warning("Please provide all required parameters (API key, model, prompt, and chunks) to enable processing.")


if __name__ == "__main__":
    main()
