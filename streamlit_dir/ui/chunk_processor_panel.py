import logging
from datetime import datetime
from typing import Any

import streamlit as st

from model.core.chunk.chunk_manager import ChunkManager
from model.io.gemini_result_saver import GeminiResultSaver
from model.utils.result_type import ResultType
from streamlit_dir.gemini_chunk_processor import GeminiChunkProcessor
from streamlit_dir.ui.token_usage_gauge import render_token_usage_gauge

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def render_status_panel(processor, chunk_manager, processed, total_chunks, total_tokens):
    """Unified display of chunk progress, token usage, and stats."""
    render_progress_with_info("Chunks Processed", processed, total_chunks)
    remaining_ratio = processor.remaining_tokens / total_tokens * 100
    render_token_usage_gauge(remaining_ratio)


def render_progress_with_info(label: str, current: int, total: int, icon: str = "📦"):
    """Renders a progress bar and a stat info block below it."""
    st.progress(current / total, text=f"{label}: {current}/{total}")
    st.info(f"{icon} {label}: {total} 🔁 Total: {total}")


def process_chunks_ui(
    client: Any,
    prompt: str,
    chunk_file_path: str,
    chunk_count: int,
    total_tokens: int
):
    st.markdown("### 🧠 Chunk Processing Progress")

    if not all([client, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    # Load manager & processor
    chunk_manager = ChunkManager(json_path=chunk_file_path)
    processor = GeminiChunkProcessor(client=client, prompt=prompt, chunk_manager=chunk_manager)

    start = st.button("Start Processing")

    # placeholders for live updates
    update_area = st.empty()

    if start:
        results = []
        processed = 0

        for i in range(chunk_count):
            try:
                result = processor.process_one_chunk()
            except Exception as e:
                update_area.error(f"❌ Exception: {e}", icon="🚨")
                logger.exception("Unexpected exception during chunk processing")
                break

            if result.result_type == ResultType.SUCCESS:
                results.append(result)
                processed += 1

            elif result.result_type == ResultType.FATAL_ERROR:
                update_area.error(f"❌ Fatal Error: {result.error}", icon="🚨")
                with update_area.container():
                    render_status_panel(processor, chunk_manager, processed, chunk_count, total_tokens)
                break

            elif result.result_type == ResultType.RETRYABLE_ERROR:
                update_area.warning(f"⚠️ Retryable Error: {result.error}", icon="🔁")

            elif result.result_type == ResultType.NO_MORE_CHUNKS:
                update_area.info("✅ No more chunks to process.", icon="📭")
                break

            else:
                update_area.error(f"❓ Unexpected Error: {result.error}", icon="❓")

            # UI updates after every step
            with update_area.container():
                render_status_panel(processor, chunk_manager, processed, chunk_count, total_tokens)

        else:
            with update_area.container():
                render_status_panel(processor, chunk_manager, processed, chunk_count, total_tokens)
                st.success("✅ Finished processing all requested chunks.")

    else:
        update_area.info("ℹ️ Click 'Start Processing' to begin.", icon="🟢")
