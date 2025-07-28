import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import time
import dotenv
import pandas as pd
import streamlit as st

from model.core.chunk.chunk_json_inspector import ChunkJSONInspector
from model.core.chunk.chunker import DataFrameChunker
from model.core.llms.gemini_model_provider import GeminiModelProvider
from model.io.model_prefs import ModelPreference
from model.utils.constants import TEMP_DIR
from streamlit_dir.prompt_pref import PromptPreference
from streamlit_dir.stramlit_dataset_handler import StreamlitDatasetHandler


@st.cache_data(show_spinner="🔍 Fetching available Gemini models...")
def get_available_models(api_key: str):
    provider = GeminiModelProvider(api_key)
    return provider.get_usable_model_names()

def load_api_key_ui(container) -> str:
    dotenv.load_dotenv()
    saved_key = os.getenv("KEY")

    api_key = container.text_input(
        "🔑 Enter your Gemini API Key",
        value=saved_key if saved_key else "",
        type="password",
        help="Stored in your .env file"
    )

    if api_key and api_key != saved_key:
        if container.checkbox("💾 Save this key"):
            dotenv.set_key(dotenv.find_dotenv(), "KEY", api_key)
            container.success("✅ API key saved")
    elif api_key:
        container.info("✅ Using Saved API key")

    return api_key


def model_selector_ui(container, api_key: str) -> str:
    model_pref = ModelPreference()
    saved_model = model_pref.get_model_name()

    if saved_model:
        container.info(f"✅ Using saved model: `{saved_model}`")
        fetch = container.checkbox("🔄 Fetch new model list?", value=False)
        if not fetch:
            return saved_model

    model_names = get_available_models(api_key)

    if not model_names:
        container.error("❌ No usable models found. Please check your API key.")
        st.stop()

    selected_model = container.selectbox("🧠 Select a Gemini model", model_names)

    if selected_model != saved_model:
        model_pref.save_model_name(selected_model)
        container.success(f"✅ Model `{selected_model}` saved.")

    return selected_model


def prompt_input_ui(container):
    prompt_pref = PromptPreference()
    saved_prompt = prompt_pref.load_prompt()

    prompt = container.text_area("💬 Enter your prompt", value=saved_prompt, height=200)

    if prompt and prompt != saved_prompt:
        if container.button("💾 Save Prompt"):
            prompt_pref.save_prompt(prompt)
            container.success("✅ Prompt saved")

    return prompt

def chunk_and_save_dataframe(df: pd.DataFrame, chunk_size: int) -> dict:
    save_path = "temp\\chunks.json"  # Save to working dir
    chunker = DataFrameChunker(chunk_size)
    chunks = chunker.chunk_dataframe(df)
    chunker.save_chunks_to_json(chunks, file_path=save_path)

    # Use inspector to extract all summary info
    inspector = ChunkJSONInspector(directory_path=".")
    summary = inspector.inspect_chunk_file(Path(save_path))

    return {
        "chunk_file_path": save_path,
        "summary": summary
    }

def handle_dataset_upload_or_load_and_chunk() -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Dict]]:
    handler = StreamlitDatasetHandler()

    saved_filename = st.session_state.get("saved_filename") or handler.get_saved_file_name()
    df = None
    chunk_summary = None
    chunk_file_path = None

    if saved_filename and not st.session_state.get("upload_new_file"):
        st.success(f"📁 Using saved file: `{saved_filename}`")

        if st.button("📤 Upload a new file?"):
            st.session_state["upload_new_file"] = True
            st.rerun()

        file_path = handler.save_dir / saved_filename  # Use the full path from save_dir
        if file_path.exists():
            try:
                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path)
                elif file_path.suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    st.error("❌ Unsupported file format.")
            except Exception as e:
                st.error(f"❌ Failed to load file: {e}")
        else:
            st.error(f"❌ File not found: {file_path}")
    else:
        uploaded_file = st.file_uploader("📂 Upload CSV or Parquet", type=["csv", "parquet"])
        df = handler.load_from_upload(uploaded_file)

        if df is not None:
            if st.button("💾 Save file to disk"):
                saved_path = handler.save_uploaded_file()
                st.session_state["saved_filename"] = saved_path
                st.session_state["upload_new_file"] = False
                st.success(f"✅ File saved: `{saved_path}`")
                st.rerun()

    # --- Separator ---
    st.markdown("---")

    # --- Chunking UI ---
    if df is not None:
        chunk_size = st.number_input("🔢 Set Chunk Size", min_value=1, value=100)
        if st.button("📦 Chunk & Save"):
            result = chunk_and_save_dataframe(df, chunk_size)
            chunk_file_path = result["chunk_file_path"]
            chunk_summary = result["summary"]
            st.success(f"✅ Chunks saved to: `{chunk_file_path}`")

        # If a summary already exists, show it
        if chunk_summary:
            st.subheader("📊 Chunk Summary")
            for k, v in chunk_summary.items():
                st.write(f"**{k.replace('_', ' ').capitalize()}:** {v}")

    return df, saved_filename, chunk_summary
