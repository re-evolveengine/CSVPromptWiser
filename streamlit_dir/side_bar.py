import streamlit as st
from model.utils.constants import APP_NAME
from streamlit_dir.side_bar_utils import model_selector_ui, load_api_key_ui
from streamlit_dir.stramlit_dataset_handler import StreamlitDatasetHandler


def cwp_sidebar():
    st.sidebar.header(f"{APP_NAME} Controls")

    # Step 0: Load API Key and Fetch Available Models
    api_key = load_api_key_ui()

    selected_model = model_selector_ui(api_key)

    # # File upload (CSV/Parquet)
    handler = StreamlitDatasetHandler()

    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV or Parquet", type=["csv", "parquet"])
    df = handler.load_from_upload(uploaded_file)

    if df is not None:
        st.success("✅ Dataset loaded successfully!")
        st.dataframe(df.head())

        if st.button("💾 Save file to disk"):
            saved_path = handler.save_uploaded_file()
            st.info(f"File saved to: `{saved_path}`")

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

