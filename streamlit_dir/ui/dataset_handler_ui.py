import streamlit as st
from pathlib import Path
import os
from typing import Optional, Dict, Tuple
import pandas as pd

from model.core.chunk.chunk_json_inspector import ChunkJSONInspector
from model.core.chunk.chunker import DataFrameChunker
from streamlit_dir.stramlit_dataset_handler import StreamlitDatasetHandler


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

    return df, saved_filename, chunk_summary