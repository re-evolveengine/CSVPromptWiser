import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from model.core.chunk.chunk_manager import ChunkManager
from model.io.gemini_result_saver import GeminiResultSaver
from model.utils.constants import RESULTS_DIR
from model.utils.result_type import ResultType
from streamlit_dir.gemini_chunk_processor import GeminiChunkProcessor
from streamlit_dir.ui.token_usage_gauge import render_token_usage_gauge

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def render_status(processor, chunk_manager, processed, total):
    st.info(f"🪙 Remaining Tokens: {processor.remaining_tokens}")
    st.info(f"📦 Remaining Chunks: {chunk_manager.remaining_chunks}")
    st.info(f"✅ Chunks Processed This Run: {processed}/{total}")


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

    # UI setup
    st.info(f"📦 Total Chunks: {chunk_manager.total_chunks} 🔁 Remaining: {chunk_manager.remaining_chunks}")
    start = st.button("Start Processing")

    # placeholders for live updates
    update_area = st.empty()

    if start:
        results = []
        processed = 0

        for i in range(chunk_count):
            # 1) call out to the API
            try:
                result = processor.process_one_chunk()
            except Exception as e:
                # catch unexpected exceptions to avoid breaking the loop completely
                update_area.error(f"❌ Exception: {e}", icon="🚨")
                logger.exception("Unexpected exception during chunk processing")
                break

            # 2) handle result types
            if result.result_type == ResultType.SUCCESS:
                results.append(result)
                processed += 1

            elif result.result_type == ResultType.FATAL_ERROR:
                update_area.error(f"❌ Fatal Error: {result.error}", icon="🚨")
                render_status(processor, chunk_manager, processed, chunk_count)
                break

            elif result.result_type == ResultType.RETRYABLE_ERROR:
                update_area.warning(f"⚠️ Retryable Error: {result.error}", icon="🔁")
                # optional: implement backoff / retry logic here

            elif result.result_type == ResultType.NO_MORE_CHUNKS:
                update_area.info("✅ No more chunks to process.", icon="📭")
                break

            else:
                update_area.error(f"❓ Unexpected Error: {result.error}", icon="❓")

            # 3) render live status in the same spot
            with update_area.container():
                st.progress(processed / chunk_count)
                render_status(processor, chunk_manager, processed, chunk_count)

        else:
            # only runs if loop didn't break
            with update_area.container():
                render_status(processor, chunk_manager, processed, chunk_count)
                st.success("✅ Finished processing all requested chunks.")

        # 4) save results if any
        # if results:
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     base = f"{client.model}_{timestamp}"
        #     json_path = Path(RESULTS_DIR) / f"{base}.json"
        #     csv_path = Path(RESULTS_DIR) / f"{base}.csv"
        #
        #     GeminiResultSaver.save_results_to_json(results, str(json_path))
        #     GeminiResultSaver.save_results_to_csv(results, str(csv_path))
        #     st.success("✅ Results saved successfully.")

    else:
        # idle state
        update_area.info("ℹ️ Click 'Start Processing' to begin.", icon="🟢")

    # Token usage gauge always visible
    st.subheader("🧮 Token Usage Summary")
    used_ratio = (total_tokens - processor.remaining_tokens) / total_tokens
    render_token_usage_gauge(used_ratio * 100)
