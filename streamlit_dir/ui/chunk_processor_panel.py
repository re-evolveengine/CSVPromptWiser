import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from model.core.chunk.chunk_manager import ChunkManager
from model.io.gemini_result_saver import GeminiResultSaver
from model.utils.constants import RESULTS_DIR
from model.utils.result_type import ResultType
from streamlit_dir.chunk_process_state import CurrentChunkState
from streamlit_dir.gemini_chunk_processor import GeminiChunkProcessor
from streamlit_dir.ui.token_usage_gauge import render_token_usage_gauge

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def render_status_info(processor: GeminiChunkProcessor, chunk_manager: ChunkManager, state: CurrentChunkState):
    st.info(
        f"🪙 Remaining Tokens: {processor.remaining_tokens}",
        key="remaining_tokens_info"
    )
    st.info(
        f"📦 Remaining Chunks: {chunk_manager.remaining_chunks}",
        key="remaining_chunks_info"
    )
    st.info(
        f"✅ Chunks Processed This Run: {state.processed_chunk_count}",
        key="processed_chunk_info"
    )


def process_chunks_ui(
    client: Any,
    prompt: str,
    chunk_file_path: str,
    chunk_count: int,
    total_tokens: int
):
    """
    Streamlit panel to process chunks with a pre-configured GeminiClient.
    """

    st.markdown("### 🧠 Chunk Processing Progress")

    if not all([client, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    # Load chunk data
    chunk_manager = ChunkManager(json_path=chunk_file_path)
    total_chunks = chunk_manager.total_chunks
    remaining_chunks = chunk_manager.remaining_chunks

    # Initialize session state if needed
    if "chunk_state" not in st.session_state:
        st.session_state.chunk_state = CurrentChunkState(
            total_chunk_count=chunk_count,
            processed_chunk_count=0
        )

    state: CurrentChunkState = st.session_state.chunk_state

    # Setup Gemini processor
    processor = GeminiChunkProcessor(
        client=client,
        prompt=prompt,
        chunk_manager=chunk_manager
    )

    # UI placeholders
    st.info(f"📦 Total Chunks: {total_chunks} 🔁 Remaining: {remaining_chunks}", key="initial_chunk_stats")
    progress_bar = st.progress(0, text="Waiting to start...", key="chunk_progress")

    # Start processing if triggered
    if st.session_state.get("start_processing", False):
        results = []

        for _ in range(state.total_chunk_count):
            result = processor.process_one_chunk()

            if result.result_type == ResultType.SUCCESS:
                results.append(result)
                state.processed_chunk_count += 1

            elif result.result_type == ResultType.FATAL_ERROR:
                st.error(f"❌ Fatal Error on Chunk: {result.error}", icon="🚨")
                render_status_info(processor, chunk_manager, state)
                break

            elif result.result_type == ResultType.RETRYABLE_ERROR:
                st.warning(f"⚠️ Retryable Error: {result.error}", icon="🔁")

            elif result.result_type == ResultType.UNEXPECTED_ERROR:
                st.error(f"❌ Unexpected Error: {result.error}", icon="❓")

            elif result.result_type == ResultType.NO_MORE_CHUNKS:
                st.info("✅ All chunks have already been processed.", icon="📭")
                break

            # Live status inside loop
            progress = (state.processed_chunk_count / state.total_chunk_count)
            progress_bar.progress(progress, text=f"{state.processed_chunk_count} / {state.total_chunk_count}")
            render_status_info(processor, chunk_manager, state)

        else:
            # If loop completes successfully
            render_status_info(processor, chunk_manager, state)
            st.success("✅ Finished processing all requested chunks.")

        # Save results
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{client.model}_{timestamp}"
            json_path = Path(RESULTS_DIR) / f"{base}.json"
            csv_path = Path(RESULTS_DIR) / f"{base}.csv"

            GeminiResultSaver.save_results_to_json(results, str(json_path))
            GeminiResultSaver.save_results_to_csv(results, str(csv_path))

            st.success("✅ Results saved successfully.")

        # Reset processing trigger and rerun
        st.session_state["start_processing"] = False
        st.rerun()

    else:
        # Idle state
        progress_bar.empty()
        st.info("ℹ️ Click 'Start Processing' to begin.", icon="🟢")

    # 🎯 Token gauge
    st.subheader("🧮 Token Usage Summary")
    used_ratio = (total_tokens - processor.remaining_tokens) / total_tokens
    render_token_usage_gauge(used_ratio * 100)
