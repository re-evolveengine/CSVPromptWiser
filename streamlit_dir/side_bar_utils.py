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
    saved_models = model_pref.get_model_list()
    saved_selected_model = model_pref.get_selected_model_name()
    selected_model = None

    # Step 1: Check if previously selected model exists and prompt to use it
    if saved_selected_model:
        container.markdown("### 🧠 Model Selection")
        container.info(f"📌 A previously selected model was found: `{saved_selected_model}`")
        use_saved = container.button("✅ Use saved model", key="use_saved_model")
        if use_saved:
            return saved_selected_model

    # Step 2: If user chose not to use saved model or none exists, show saved model list if available
    if saved_models and (not saved_selected_model or not use_saved):
        if not container.checkbox("📌 Show previously used models", value=True):
            saved_models = []

    # Step 3: If no saved models or user wants to fetch new ones, fetch from API
    fetch_new = not saved_models or container.checkbox("🔄 Fetch latest model list from API", value=not bool(saved_models))
    
    if fetch_new:
        with container.status("🔍 Fetching available models..."):
            model_names = get_available_models(api_key)
            if not model_names:
                container.error("❌ No usable models found. Please check your API key.")
                st.stop()
            model_pref.save_model_list(model_names)
    else:
        model_names = saved_models

    # Let user select from the available models
    if not model_names:
        container.error("❌ No models available to display.")
        st.stop()

    selected_model = container.selectbox(
        "🧠 Select a Gemini model",
        model_names,
        index=model_names.index(saved_selected_model) if saved_selected_model in model_names else 0
    )
    # hey ChatGPT. IT SHOULD BE HERE

    # Save selected model if changed
    if selected_model != saved_selected_model:
        model_pref.save_selected_model_name(selected_model)
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
    os.makedirs(TEMP_DIR, exist_ok=True)
    save_path = os.path.join(TEMP_DIR, "chunks.json")

    chunker = DataFrameChunker(chunk_size)
    chunks = chunker.chunk_dataframe(df)
    chunker.save_chunks_to_json(chunks, file_path=save_path)

    # Inspect the chunk summary
    inspector = ChunkJSONInspector(directory_path=TEMP_DIR)
    summary = inspector.inspect_chunk_file(Path(save_path))

    return {
        "chunk_file_path": save_path,
        "summary": summary
    }

from model.utils.constants import TEMP_DIR

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

        file_path = handler.save_dir / saved_filename
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

    # --- Load chunk summary if exists ---
    chunk_file = Path(TEMP_DIR) / "chunks.json"
    if chunk_file.exists():
        try:
            inspector = ChunkJSONInspector(directory_path=TEMP_DIR)
            chunk_summary = inspector.inspect_chunk_file(chunk_file)
        except Exception as e:
            st.warning(f"⚠️ Failed to read chunk summary: {e}")

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

    # --- Show chunk summary (if available) ---
    if chunk_summary:
        st.subheader("📊 Chunk Summary")
        for k, v in chunk_summary.items():
            st.write(f"**{k.replace('_', ' ').capitalize()}:** {v}")

    return df, saved_filename, chunk_summary

