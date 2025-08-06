import logging
from typing import Any
from datetime import datetime

import streamlit as st

from streamlit_dir.chunk_process_state import ChunkProcessorState
from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.utils.result_type import ResultType
from streamlit_dir.gemini_chunk_processor import GeminiChunkProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def process_chunks_ui(
    client: Any,
    prompt: str,
    chunk_file_path: str,
    chunk_count: int,
    total_tokens: int
):
    """
    Streamlit panel to process multiple chunks using GeminiChunkProcessor.
    """

    st.markdown("### 🧠 Chunk Processing Progress")

    if not all([client, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    if "chunk_state" not in st.session_state:
        st.session_state.chunk_state = ChunkProcessorState(total_tokens=total_tokens)

    state: ChunkProcessorState = st.session_state.chunk_state

    chunk_manager = ChunkManager(json_path=chunk_file_path)
    total_chunks = chunk_manager.total_chunks
    remaining_chunks = chunk_manager.remaining_chunks

    st.info(f"📦 Total Chunks: {total_chunks} 🔁 Remaining: {remaining_chunks}")

    processor = GeminiChunkProcessor(prompt=prompt, client=client, chunk_manager=chunk_manager)

    if "start_processing" in st.session_state and st.session_state.get("start_processing"):
        results = []
        errors = []

        for i in range(chunk_count):
            result = processor.process_one_chunk()

            state.processed_chunk_count = i + 1
            state.chunk_count = chunk_count
            state.remaining_tokens = result.remaining_tokens or state.remaining_tokens

            if result.result_type == ResultType.SUCCESS:
                results.append(result)
            elif result.result_type == ResultType.FATAL_ERROR:
                errors.append(result.error)
                break
            elif result.result_type in (ResultType.RETRYABLE_ERROR, ResultType.UNEXPECTED_ERROR):
                errors.append(result.error)
                continue
            elif result.result_type == ResultType.NO_MORE_CHUNKS:
                break

        if errors:
            st.error("⚠️ Some chunks failed:")
            for err in errors:
                st.write(f"- {err}")

        if not results:
            st.error("❌ No chunks were processed successfully.")
        else:
            st.success("✅ Processing complete.")

        st.session_state["start_processing"] = False
        st.rerun()
    else:
        st.info("ℹ️ Click 'Start Processing' to begin chunk processing.")

    st.subheader("🧮 Token Usage Summary")
    current_percent = ((state.total_tokens - state.remaining_tokens) / state.total_tokens) * 100 if state.total_tokens else 0
    st.progress(current_percent / 100.0)
    st.info(f"🪙 Remaining Tokens So Far: {state.remaining_tokens}")
