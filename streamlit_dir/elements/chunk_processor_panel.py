import logging

import streamlit as st

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.io.gemini_sqlite_result_saver import GeminiSQLiteResultSaver
from model.io.model_prefs import ModelPreference
from model.utils.result_type import ResultType
from model.io.save_processed_chunks_to_db import save_processed_chunk_to_db
from model.core.chunk.gemini_chunk_processor import GeminiChunkProcessor
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
    st.info(f"{icon} {label}: {total} 🔁 Total: {total}")


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
    processor = GeminiChunkProcessor(client=client, prompt=prompt, chunk_manager=chunk_manager)
    prefs = ModelPreference()

    start = st.button("Start Processing")

    # placeholders for live updates
    update_area = st.empty()

    if start:
        results = []
        processed = 0

        for i in range(chunk_count):
            try:
                result = processor.process_next_chunk()
            except Exception as e:
                update_area.error(f"❌ Exception: {e}", icon="🚨")
                logger.exception("Unexpected exception during chunk processing")
                break

            if result.result_type == ResultType.SUCCESS:
                results.append(result)

                save_processed_chunk_to_db(
                    result=result,
                    chunk_id=result.chunk_id,
                    prompt=prompt,
                    model_version=client.model_name,
                    saver=GeminiSQLiteResultSaver()
                )

                processed += 1

            elif result.result_type == ResultType.FATAL_ERROR:
                update_area.error(f"❌ Fatal Error: {result.error}", icon="🚨")
                with update_area.container():
                    render_status_panel(chunk_manager, prefs, processed, chunk_count)
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
                render_status_panel(chunk_manager, prefs, processed, chunk_count)

        else:
            with update_area.container():
                render_status_panel(chunk_manager, prefs, processed, chunk_count)
                st.success("✅ Finished processing all requested chunks.")

    else:
        update_area.info("ℹ️ Click 'Start Processing' to begin.", icon="🟢")
