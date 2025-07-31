import streamlit as st
from model.utils.constants import APP_NAME
from streamlit_dir.side_bar import cwp_sidebar
from streamlit_dir.ui.chunk_processor_panel import process_chunks_ui

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    initial_sidebar_state="expanded",
    layout="wide"
)

st.markdown(
    """
    <style>
        /* Main app container */
        .stApp {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        /* Main content area */
        .main .block-container {
            max-width: 100% !important;
            padding: 1rem 1rem 1rem 1rem !important;
        }

        /* Content wrapper */
        .main > div {
            max-width: 100% !important;
        }

        /* Text and code blocks */
        .stMarkdown, .stText, .stCodeBlock, .stDataFrame {
            max-width: 100% !important;
        }

        /* Code and text areas */
        .stCodeBlock pre, .stTextArea textarea, pre, code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            max-width: 100% !important;
        }

        /* Tables */
        .stDataFrame {
            display: block;
            overflow-x: auto;
            width: 100% !important;
        }

        /* Fix for Streamlit's dynamic content */
        [data-testid="stAppViewContainer"] > .main > div {
            padding: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)




def main():

    st.title(f"🤖 {APP_NAME} Dashboard")

    # --- Sidebar interaction ---
    api_key, model_name, df, chunk_file_path, chunk_summary, prompt, generation_config, gemini_client = cwp_sidebar()

    # --- Chunk Processing Output ---
    if "start_processing" in st.session_state and st.session_state.get("start_processing"):
        try:
            process_chunks_ui(
                gemini_client,
                prompt,
                chunk_file_path,
                max_chunks=st.session_state.get("num_chunks", 5)
            )
        finally:
            # Reset the processing state when done
            st.session_state["start_processing"] = False
            st.rerun()  # Rerun to update the UI

    # --- Model Configuration ---
    if model_name and generation_config:
        st.subheader("⚙️ Model Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", model_name)
        with col2:
            st.metric("Temperature", f"{generation_config.get('temperature', 0.2):.2f}")
        with col3:
            st.metric("Top K", generation_config.get('top_k', 40))
        with col4:
            st.metric("Top P", f"{generation_config.get('top_p', 1.0):.2f}")

    # --- Chunk Summary ---
    if chunk_summary:
        st.subheader("📦 Chunk Summary")
        for key, value in chunk_summary.items():
            st.markdown(f"- **{key.replace('_', ' ').capitalize()}**: {value}")

    # --- Dataset Preview ---
    if df is not None:
        st.subheader("📊 Dataset Preview")
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(5), use_container_width=True)

    # --- Prompt Echo ---
    if prompt:
        st.subheader("💬 Your Prompt")
        st.code(prompt, language="markdown")

if __name__ == "__main__":
    main()
