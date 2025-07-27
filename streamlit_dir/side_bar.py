import streamlit as st
from model.utils.constants import APP_NAME
from streamlit_dir.side_bar_utils import model_selector_ui, load_api_key_ui
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
        handler = StreamlitDatasetHandler()
        uploaded_file = upload_section.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
        df = handler.load_from_upload(uploaded_file)

        if df is not None:
            st.success("✅ Dataset loaded successfully!")
            st.dataframe(df.head())

            if st.button("💾 Save file to disk"):
                saved_path = handler.save_uploaded_file()
                st.info(f"File saved to: `{saved_path}`")

    return api_key, selected_model, uploaded_file, df


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
