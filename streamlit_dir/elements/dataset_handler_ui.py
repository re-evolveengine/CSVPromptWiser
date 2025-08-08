import streamlit as st
from pathlib import Path
import os
from typing import Optional, Dict, Tuple
import pandas as pd

from model.core.chunk.chunk_json_inspector import ChunkJSONInspector
from model.core.chunk.chunker import DataFrameChunker
from model.io.model_prefs import ModelPreference
from model.io.dataset_handler import DatasetHandler
from model.core.llms.prompt_optimizer import PromptOptimizer
from utils.constants import TEMP_DIR, DEFAULT_TOKEN_BUDGET


def chunk_and_save_dataframe(df: pd.DataFrame, chunk_size: int) -> dict:
    os.makedirs(TEMP_DIR, exist_ok=True)
    save_path = os.path.join(TEMP_DIR, "chunks.json")

    chunker = DataFrameChunker(chunk_size)
    chunks = chunker.chunk_dataframe(df)
    chunker.save_chunks_to_json(chunks, file_path=save_path)

    inspector = ChunkJSONInspector(directory_path=TEMP_DIR)
    summary = inspector.inspect_chunk_file(Path(save_path))

    return {
        "chunk_file_path": save_path,
        "summary": summary
    }


def handle_dataset_upload_or_load() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Handle file upload or load existing file.
    
    Returns:
        Tuple containing (dataframe, saved_filename)
    """
    handler = DatasetHandler()
    saved_filename = st.session_state.get("saved_filename") or handler.get_saved_file_name()
    df = None

    if saved_filename and not st.session_state.get("upload_new_file"):
        st.success(f"📁 Using saved file: `{saved_filename}")

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
    
    return df, saved_filename


def configure_and_process_chunks(df: pd.DataFrame,prompt:str,response_example:str, optimizer: Optional[PromptOptimizer] = None) -> Tuple[Optional[str], Optional[Dict], Optional[int]]:
    """Configure chunking settings and process the dataframe into chunks.

    Args:
        df: The dataframe to chunk
        optimizer: Optional PromptOptimizer for optimal chunk size calculation

    Returns:
        Tuple containing (chunk_file_path, chunk_summary)
    """
    chunk_summary = None
    chunk_file_path = None

    # Load existing chunk summary
    chunk_file = Path(TEMP_DIR) / "chunks.json"
    if chunk_file.exists():
        try:
            inspector = ChunkJSONInspector(directory_path=TEMP_DIR)
            chunk_summary = inspector.inspect_chunk_file(chunk_file)
            chunk_file_path = str(chunk_file)
        except Exception as e:
            st.warning(f"⚠️ Failed to read chunk summary: {e}")

    st.markdown("### Chunking Settings")

    # Show optimal chunk size
    if optimizer is not None and len(df) > 0:
        try:
            optimal_size = optimizer.find_optimal_row_number(
                prompt=prompt,
                row_df=df.head(1),
                example_response=response_example
            )
            st.info(f"ℹ️ Recommended chunk size: **{optimal_size}** rows (based on model context window)")
        except Exception as e:
            st.warning(f"⚠️ Could not calculate optimal chunk size: {str(e)}")
    else:
        st.info("ℹ️ Connect a model to see recommended chunk size")

    prefs = ModelPreference()

    token_budget = st.number_input(
        "Enter Token Budget",
        min_value=1,
        value=prefs.total_tokens,
    )
    prefs.total_tokens = token_budget
    prefs.remaining_total_tokens = token_budget

    # Chunk size input
    chunk_size = st.number_input(
        "🔢 Set Chunk Size",
        min_value=1,
        value=50,
        help="Number of rows per chunk. Consider the recommended size above for optimal performance."
    )

    # Show tokens per chunk and number of chunks possible
    if optimizer is not None and len(df) > 0:
        try:
            tokens_per_chunk = optimizer.calculate_used_tokens(
                prompt=prompt,
                row_df=df.head(1),
                example_response=response_example,
                num_rows=chunk_size
            )

            if tokens_per_chunk > 0:
                max_chunks = token_budget // tokens_per_chunk
                st.caption(f"📊 Estimated tokens per chunk: **{tokens_per_chunk:,}**")
                st.success(f"🔢 You can process approximately **{max_chunks} chunks** with your budget.")
            else:
                st.warning("⚠️ Could not estimate tokens per chunk.")
        except Exception as e:
            st.warning(f"⚠️ Error estimating chunks: {e}")

    if st.button("📦 Chunk & Save"):
        result = chunk_and_save_dataframe(df, chunk_size)
        chunk_file_path = result["chunk_file_path"]
        chunk_summary = result["summary"]
        st.success(f"✅ Chunks saved to: `{chunk_file_path}`")

    return chunk_file_path, chunk_summary, token_budget


def configure_and_process_chunks(df: pd.DataFrame, prompt: str, response_example: str,
                                 optimizer: Optional[PromptOptimizer] = None) -> Tuple[
    Optional[str], Optional[Dict], Optional[int]]:
    """Configure chunking settings and process the dataframe into chunks.

    Args:
        df: The dataframe to chunk
        optimizer: Optional PromptOptimizer for optimal chunk size calculation

    Returns:
        Tuple containing (chunk_file_path, chunk_summary)
    """
    # --- MODIFICATION START: Retrieve results from session_state ---
    # This ensures that after a rerun (from the dialog), the values persist.
    chunk_file_path = st.session_state.get("chunk_file_path", None)
    chunk_summary = st.session_state.get("chunk_summary", None)
    # --- MODIFICATION END ---

    # Load existing chunk summary if the path isn't already in session state
    if not chunk_file_path:
        chunk_file = Path(TEMP_DIR) / "chunks.json"
        if chunk_file.exists():
            try:
                inspector = ChunkJSONInspector(directory_path=TEMP_DIR)
                chunk_summary = inspector.inspect_chunk_file(chunk_file)
                chunk_file_path = str(chunk_file)
                st.session_state.chunk_file_path = chunk_file_path
                st.session_state.chunk_summary = chunk_summary
            except Exception as e:
                st.warning(f"⚠️ Failed to read chunk summary: {e}")

    st.markdown("### Chunking Settings")

    # Show optimal chunk size
    if optimizer is not None and len(df) > 0:
        try:
            optimal_size = optimizer.find_optimal_row_number(
                prompt=prompt,
                row_df=df.head(1),
                example_response=response_example
            )
            st.info(f"ℹ️ Recommended chunk size: **{optimal_size}** rows (based on model context window)")
        except Exception as e:
            st.warning(f"⚠️ Could not calculate optimal chunk size: {str(e)}")
    else:
        st.info("ℹ️ Connect a model to see recommended chunk size")

    prefs = ModelPreference()
    token_budget = st.number_input(
        "Enter Token Budget", min_value=1, value=prefs.total_tokens
    )
    prefs.total_tokens = token_budget
    prefs.remaining_total_tokens = token_budget

    chunk_size = st.number_input(
        "🔢 Set Chunk Size", min_value=1, value=50, help="Number of rows per chunk."
    )

    if optimizer is not None and len(df) > 0:
        try:
            tokens_per_chunk = optimizer.calculate_used_tokens(
                prompt=prompt,
                row_df=df.head(1),
                example_response=response_example,
                num_rows=chunk_size
            )
            if tokens_per_chunk > 0:
                max_chunks = token_budget // tokens_per_chunk
                st.caption(f"📊 Estimated tokens per chunk: **{tokens_per_chunk:,}**")
                st.success(f"🔢 You can process approximately **{max_chunks} chunks** with your budget.")
        except Exception as e:
            st.warning(f"⚠️ Error estimating chunks: {e}")

    # --- MODIFICATION START: Define the callback and handle the button logic ---

    # 1. Define the action to perform as a callback function.
    def chunking_action():
        with st.spinner("Chunking dataset..."):
            result = chunk_and_save_dataframe(df, chunk_size)
        # Save results to session_state so they are not lost on rerun
        st.session_state.chunk_file_path = result["chunk_file_path"]
        st.session_state.chunk_summary = result["summary"]
        st.success(f"✅ Chunks saved to: `{st.session_state.chunk_file_path}`")

    # 2. When the button is clicked, check the database.
    if st.button("📦 Chunk & Save"):
        db_saver = GeminiSQLiteResultSaver()
        if db_saver.has_results():
            # If results exist, set a flag to show the warning dialog.
            st.session_state.show_chunking_warning = True
        else:
            # Otherwise, perform the chunking action immediately.
            chunking_action()

    # 3. Call your existing dialog function, passing the action as the callback.
    # The dialog will only render if the session_state flag is True.
    if 'show_chunking_warning' not in st.session_state:
        st.session_state.show_chunking_warning = False

    render_chunking_warning_dialog(on_confirm_callback=chunking_action)

    # --- MODIFICATION END ---

    # Return the values, which may have been updated from session_state
    return st.session_state.get("chunk_file_path"), st.session_state.get("chunk_summary"), token_budget


def handle_dataset_upload_or_load_and_chunk(optimizer: Optional[PromptOptimizer] = None) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str], Optional[Dict]]:
    """Handle dataset upload/load and chunking process.
    
    This is a wrapper function that combines the file handling and chunking operations.
    """
    df, saved_filename = handle_dataset_upload_or_load()
    chunk_file_path, chunk_summary = (None, None)
    
    if df is not None:
        chunk_file_path, chunk_summary = configure_and_process_chunks(df, optimizer)
    
    st.markdown("---")
    return df, saved_filename, chunk_file_path, chunk_summary
