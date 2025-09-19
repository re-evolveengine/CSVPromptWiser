import logging

import streamlit as st

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.io.gemini_sqlite_result_saver import SQLiteResultSaver
from model.io.model_prefs import ModelPreference
from streamlit_dir.elements.render_export_section import render_export_section
from utils.result_type import ResultType
from model.io.save_processed_chunks_to_db import save_processed_chunk_to_db
from model.core.chunk.chunk_processor import ChunkProcessor
from streamlit_dir.elements.token_usage_gauge import render_token_usage_gauge

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def remaining_to_processed(remaining: int, total: int) -> int:
    return total - remaining


def render_status_panel(
    chunk_manager: ChunkManager,
    model_prefs: ModelPreference,
    curr_processed_chunks: int,
    curr_total_chunks: int
):
    """Unified display of chunk progress, token usage, and stats."""

    # === Chunks ===
    total_chunks = chunk_manager.total_chunks
    remaining_chunks = chunk_manager.remaining_chunks
    processed_chunks = remaining_to_processed(remaining_chunks, total_chunks)

    st.markdown("#### 📦 Current Session Progress")
    render_progress_with_info("🔄 Chunks Processed (This Run)", curr_processed_chunks, curr_total_chunks)

    st.markdown("#### 📊 Overall Chunk Progress")
    render_progress_with_info("🧩 Total Chunks Processed", processed_chunks, total_chunks)

    # === Tokens ===
    total_tokens = model_prefs.total_tokens
    remaining_tokens = model_prefs.remaining_total_tokens
    processed_tokens = remaining_to_processed(remaining_tokens, total_tokens)
    processed_ratio = (processed_tokens / total_tokens) * 100

    st.markdown("#### 🔋 Token Usage Overview")
    st.info(
        f"🧮 **Total Tokens:** `{total_tokens}` &nbsp;&nbsp;&nbsp; 🔁 **Remaining:** `{remaining_tokens}` &nbsp;&nbsp;&nbsp; ✅ **Consumed:** `{processed_tokens}`"
    )
    render_token_usage_gauge(processed_ratio)


def render_progress_with_info(label: str, processed: int, total: int, icon: str = "📦"):
    """Renders a progress bar and a stat info block below it."""
    st.progress(processed / total, text=f"{label}: {processed}/{total}")
    st.info(f"{icon} {label}: {processed} 🔁 Total: {total}")


def process_chunks_ui(
        client: GeminiClient,
        prompt: str,
        chunk_file_path: str,
        chunk_count: int,
):
    st.markdown("### 🧠 Chunk Processing Progress")

    if not all([client, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    # Load manager & processor
    chunk_manager = ChunkManager(json_path=chunk_file_path)
    processor = ChunkProcessor(client=client, prompt=prompt, chunk_manager=chunk_manager)
    prefs = ModelPreference()

    if "start_processing" in st.session_state and st.session_state.get("start_processing"):
        results = []
        processed = 0

        # Persistent containers for different UI sections
        progress_placeholder = st.empty()  # will overwrite each iteration
        fatal_error_box = st.container()  # persistent fatal errors
        retry_error_box = st.container()  # persistent retryable errors
        unexpected_error_box = st.container()  # persistent unexpected errors

        had_error = False

        for i in range(chunk_count):
            try:
                result = processor.process_next_chunk()
            except Exception as e:
                # This is truly unexpected — didn't go through error handling
                logger.exception("Unexpected exception during chunk processing")
                fatal_error_box.error(
                    "❌ An unexpected error occurred. Please check logs or contact support.",
                    icon="🚨"
                )
                had_error = True
                break

            if result.result_type == ResultType.SUCCESS:
                processed += 1

            elif result.result_type == ResultType.FATAL_ERROR:
                fatal_error_box.error(f"❌ Fatal Error: {result.error}", icon="🚨")
                had_error = True
                break

            elif result.result_type == ResultType.RETRYABLE_ERROR:
                retry_error_box.warning(f"⚠️ Retryable Error: {result.error}", icon="🔁")
                had_error = True

            elif result.result_type == ResultType.UNEXPECTED_ERROR:
                unexpected_error_box.error(f"❓ Unexpected Error: {result.error}", icon="❓")
                had_error = True
                break

            elif result.result_type == ResultType.NO_MORE_CHUNKS:
                # No more chunks left to process
                break

            else:
                unexpected_error_box.error(f"❓ unknown result type: {result}", icon="❓")
                had_error = True

            # Always update the live progress — only one panel at a time
            with progress_placeholder.container():
                render_status_panel(
                    chunk_manager=chunk_manager,
                    model_prefs=prefs,
                    curr_processed_chunks=processed,
                    curr_total_chunks=chunk_count
                )

        # After loop finishes
        if not had_error:
            st.success("✅ Finished processing all requested chunks.")


    else:
        st.info("ℹ️ Click 'Start Processing' to begin.", icon="🟢")
