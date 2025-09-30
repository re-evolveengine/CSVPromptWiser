import logging
import streamlit as st

from model.core.chunk.chunk_manager import ChunkManager
from model.core.chunk.chunk_processor import ChunkProcessor
from model.core.llms.gemini_client import GeminiClient
from model.io.save_processed_chunks_to_db import save_processed_chunk_to_db
from model.io.model_prefs import ModelPreference
from model.io.sqlite_result_saver import SQLiteResultSaver
from utils.providers import get_model_prefs
from streamlit_dir.elements.token_usage_gauge import render_token_usage_gauge
from utils.result_type import ResultType


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# --- Helpers ---
def remaining_to_processed(remaining: int, total: int) -> int:
    return total - remaining


def render_progress_with_info(label: str, processed: int, total: int, icon: str = "📦"):
    total = max(total, 1)  # prevent divide-by-zero crash
    st.info(f"{icon} {label}: {processed} 🔁 Total: {total}")


def render_status_panel(
    chunk_manager: ChunkManager,
    model_prefs: ModelPreference,
    curr_processed_chunks: int,
    curr_total_chunks: int,
):
    """Unified display of chunk progress, token usage, and stats."""

    # === Chunks ===
    total_chunks = chunk_manager.total_chunks
    remaining_chunks = chunk_manager.remaining_chunks
    processed_chunks = remaining_to_processed(remaining_chunks, total_chunks)

    st.markdown("##### 📦 Current Session Progress")
    render_progress_with_info(
        "🔄 Chunks Processed (This Run)", curr_processed_chunks, curr_total_chunks
    )

    st.markdown("##### 📊 Overall Chunk Progress")
    render_progress_with_info(
        "🧩 Total Chunks Processed", processed_chunks, total_chunks
    )

    # === Tokens ===
    total_tokens = model_prefs.total_tokens
    remaining_tokens = model_prefs.remaining_total_tokens
    processed_tokens = remaining_to_processed(remaining_tokens, total_tokens)
    processed_ratio = (processed_tokens / total_tokens) * 100 if total_tokens > 0 else 0

    st.markdown("##### 🔋 Token Usage Overview")
    st.info(
        f"🧮 **Total Tokens:** `{total_tokens}`"
        f" &nbsp;&nbsp;&nbsp; 🔁 **Remaining:** `{remaining_tokens}`"
        f" &nbsp;&nbsp;&nbsp; ✅ **Consumed:** `{processed_tokens}`"
    )
    render_token_usage_gauge(processed_ratio)


# --- Main UI ---
def process_chunks_ui(
    client: GeminiClient,
    prompt: str,
    chunk_file_path: str,
    chunk_count: int,
    run_now: bool = False,
):

    if not all([client, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    # Load manager & processor
    chunk_manager = ChunkManager(json_path=chunk_file_path)
    model_prefs = get_model_prefs()
    processor = ChunkProcessor(client=client, prompt=prompt, chunk_manager=chunk_manager,model_preference=model_prefs)

    # --- When not running: just show last known status once ---
    if not run_now:

        last_status = st.session_state.get(
            "last_status", {"processed": 0, "chunk_count": chunk_count}
        )

        render_status_panel(
            chunk_manager,
            model_prefs,
            last_status["processed"],
            st.session_state.get("num_chunks", chunk_count)
        )

        st.info("ℹ️ Click 'Start Processing' to begin.", icon="🟢")
        return

    # --- Running processing loop ---
    processed = 0
    had_error = False

    # Containers for error messages
    fatal_area = st.container()
    retry_area = st.container()
    unexpected_area = st.container()
    exception_area = st.container()
    token_area = st.container()

    # Status placeholder for live updates
    status_placeholder = st.empty()

    for _ in range(chunk_count):
        try:
            result = processor.process_next_chunk()
        except Exception as e:
            exception_area.error(f"❌ Exception: {e}", icon="🚨")
            logger.exception("Unexpected exception during chunk processing")
            had_error = True
            break

        if result.result_type == ResultType.SUCCESS:
            save_processed_chunk_to_db(
                result=result,
                chunk_id=result.chunk_id,
                prompt=prompt,
                model_version=client.model_name,
                saver=SQLiteResultSaver(),
            )
            # # After saving results
            st.session_state["has_results"] = True
            processed += 1

        elif result.result_type == ResultType.FATAL_ERROR:
            fatal_area.error(f"❌ Fatal Error: {result.error}", icon="🚨")
            had_error = True
            break

        elif result.result_type == ResultType.RETRYABLE_ERROR:
            retry_area.warning(f"⚠️ Retryable Error: {result.error}", icon="🔁")
            had_error = True

        elif result.result_type == ResultType.TOKENS_BUDGET_EXCEEDED:
            token_area.error("❌ Not enough tokens left.", icon="🚨")
            had_error = True
            break

        elif result.result_type == ResultType.NO_MORE_CHUNKS:
            st.info("✅ No more chunks to process.", icon="📭")
            had_error = True
            break

        elif result.result_type == ResultType.UNEXPECTED_ERROR:
            unexpected_area.error(f"❓ Unexpected Error: {result.error}", icon="❓")
            had_error = True
            break

        else:
            unexpected_area.error(f"❓ Unknown result type: {result}", icon="❓")
            had_error = True

        # Update persistent last status and render live progress
        st.session_state["last_status"] = {
            "processed": processed,
            "chunk_count": chunk_count
        }
        with status_placeholder.container():
            render_status_panel(chunk_manager, model_prefs, processed, chunk_count)

    # --- Wrap-up ---
    if not had_error:
        st.success("✅ Finished processing all requested chunks.")
        st.rerun() #This is to refresh the page to show the export section after the first run

