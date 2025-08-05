import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Any

from model.core.chunk.chunk_manager import ChunkManager
from model.io.gemini_result_saver import GeminiResultSaver
from model.io.model_prefs import ModelPreference
from model.utils.constants import RESULTS_DIR
from streamlit_dir.ui.token_usage_gauge import render_token_usage_gauge
from streamlit_dir.ui.run_gemini_chunk_processor_ui import process_next_chunk_generator


import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from model.core.chunk.chunk_manager import ChunkManager
from model.io.gemini_result_saver import GeminiResultSaver
from model.io.model_prefs import ModelPreference
from model.utils.constants import RESULTS_DIR
from streamlit_dir.ui.token_usage_gauge import render_token_usage_gauge
from streamlit_dir.ui.run_gemini_chunk_processor_ui import process_next_chunk_generator


def process_chunks_ui(
    client: Any,
    prompt: str,
    chunk_file_path: str,
    chunk_count: int,
    total_tokens: int
):
    """
    Streamlit panel to process chunks one-by-one with a GeminiClient.
    """
    logger.info("Launching chunk processing UI")
    st.markdown("### 🧠 Chunk Processing Progress")

    if not all([client, prompt, chunk_file_path]):
        logger.warning("Missing required input(s) — client: %s, prompt: %s, chunk_file_path: %s",
                       bool(client), bool(prompt), bool(chunk_file_path))
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    logger.info("Initializing ChunkManager and ModelPreference")
    chunk_manager = ChunkManager(json_path=chunk_file_path)
    prefs = ModelPreference()

    total = chunk_manager.total_chunks
    remaining = chunk_manager.remaining_chunks
    logger.info("Chunk statistics — Total: %d, Remaining: %d", total, remaining)
    st.info(f"📦 Total Chunks: {total} 🔁 Remaining: {remaining}")

    # --- Initialize UI elements once ---
    progress_bar = st.progress(0)
    st.markdown("🧮 Token Usage Summary")
    gauge_chart = st.empty()
    chunk_status = st.empty()
    error_box = st.container()
    final_status = st.empty()

    # --- Initial gauge ---
    initial_remaining = prefs.get_remaining_total_tokens()
    initial_percent = ((total_tokens - initial_remaining) / total_tokens) * 100 if total_tokens > 0 else 0
    logger.info("Initial token usage — Used: %.2f%%, Remaining: %d", initial_percent, initial_remaining)
    gauge_chart.markdown(render_token_usage_gauge(initial_percent), unsafe_allow_html=True)

    def update_status(current_idx: int, remaining_tokens: int):
        percent_used = ((total_tokens - remaining_tokens) / total_tokens) * 100 if total_tokens > 0 else 0
        return {
            "percent_used": percent_used,
            "current_index": current_idx + 1,
            "remaining_tokens": remaining_tokens
        }

    if "start_processing" in st.session_state and st.session_state.get("start_processing"):
        logger.info("Starting chunk processing loop")
        try:
            results = []
            errors = []

            generator = process_next_chunk_generator(
                prompt=prompt,
                client=client,
                chunk_manager=chunk_manager,
                chunk_count=chunk_count
            )

            for i, result in enumerate(generator):
                if result["status"] == "ok":
                    results.append(result["data"])
                    logger.info("Chunk %d processed successfully", i + 1)
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error("Chunk %d failed: %s", i + 1, error_msg)
                    errors.append(error_msg)

                status = update_status(i, result["remaining_tokens"])
                logger.debug("Progress update — Chunk: %d/%d, Tokens used: %.2f%%",
                             status["current_index"], chunk_count, status["percent_used"])

                progress_bar.progress(status["current_index"] / chunk_count)
                gauge_chart.markdown(render_token_usage_gauge(status["percent_used"]), unsafe_allow_html=True)
                chunk_status.info(f"🔄 Processing chunk {status['current_index']} of {chunk_count}")

            if errors:
                logger.warning("Completed with %d errors", len(errors))
                error_box.error("⚠️ Some chunks failed:")
                for err in errors:
                    error_box.write(f"- {err}")

            if not results:
                logger.error("No successful results to save")
                final_status.error("❌ No chunks were processed successfully.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{client.model}_{timestamp}"
            json_path = Path(RESULTS_DIR) / f"{base}.json"
            csv_path = Path(RESULTS_DIR) / f"{base}.csv"

            logger.info("Saving results to JSON: %s", json_path)
            GeminiResultSaver.save_results_to_json(results, str(json_path))
            logger.info("Saving results to CSV: %s", csv_path)
            GeminiResultSaver.save_results_to_csv(results, str(csv_path))

            logger.info("Processing complete. Results saved.")
            final_status.success("✅ Processing complete and results saved.")

        finally:
            logger.info("Cleaning up session state and rerunning UI")
            st.session_state["start_processing"] = False
            st.rerun()
    else:
        logger.debug("Waiting for user to click 'Start Processing'")
        chunk_status.info("ℹ️ Click 'Start Processing' to begin chunk processing.")

# def process_chunks_ui(
#         client: Any,
#         prompt: str,
#         chunk_file_path: str,
#         max_chunks: int,
#         total_tokens: int
# ):
#     """
#     Streamlit panel to process chunks with a pre-configured GeminiClient.
#     """
#
#     st.markdown("### 🧠 Chunk Processing Progress")
#
#     if not all([client, prompt, chunk_file_path]):
#         st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
#         return
#
#     # Init state object if not present
#     if "chunk_state" not in st.session_state:
#         st.session_state.chunk_state = ChunkProcessorState(total_tokens=total_tokens)
#
#     state: ChunkProcessorState = st.session_state.chunk_state
#
#     chunk_manager = ChunkManager(json_path=chunk_file_path)
#     total_chunks = chunk_manager.total_chunks
#     remaining_chunks = chunk_manager.remaining_chunks
#
#     # UI placeholders
#     status_container = st.empty()
#     gauge_placeholder = st.empty()
#     progress_bar = st.progress(0)
#     batch_status = st.empty()
#     status_placeholder = st.empty()
#
#     def update_status(current: int, batch_total: int, remaining_tokens: int):
#         """Callback to update processing state only."""
#         state.current_chunk = current
#         state.batch_total = batch_total
#         state.remaining_tokens = remaining_tokens
#
#     # ⏳ Display Initial Status
#     status_container.info(f"📦 Total Chunks: {total_chunks} 🔁 Remaining: {remaining_chunks}")
#     batch_status.info("⏳ Waiting to start processing...")
#
#
#     # 🔄 Processing
#     if "start_processing" in st.session_state and st.session_state.get("start_processing"):
#         try:
#             results, errors = run_gemini_chunk_processor_ui(
#                 prompt=prompt,
#                 client=client,
#                 chunk_manager=chunk_manager,
#                 max_chunks=max_chunks,
#                 progress_callback=update_status
#             )
#
#             if errors:
#                 st.error("⚠️ Some chunks failed:")
#                 for err in errors:
#                     st.write(f"- {err}")
#
#             if not results:
#                 st.error("❌ No chunks were processed successfully.")
#                 return
#
#             # ✅ Save results
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             base = f"{client.model}_{timestamp}"
#             json_path = Path(RESULTS_DIR) / f"{base}.json"
#             csv_path = Path(RESULTS_DIR) / f"{base}.csv"
#
#             GeminiResultSaver.save_results_to_json(results, str(json_path))
#             GeminiResultSaver.save_results_to_csv(results, str(csv_path))
#
#             st.success("✅ Processing complete and results saved.")
#             # Optionally: Show download links
#             # st.markdown(f"- 📁 [Download JSON Result]({json_path})")
#             # st.markdown(f"- 📁 [Download CSV Result]({csv_path})")
#
#         finally:
#             st.session_state["start_processing"] = False
#             st.rerun()
#
#     else:
#         # Idle state
#         progress_bar.empty()
#         status_placeholder.info("ℹ️ Click 'Start Processing' to begin chunk processing.")
#
#
#     # 🎯 Show token usage gauge
#     st.subheader("🧮 Token Usage Summary")
#     current_percent = ((state.total_tokens - state.remaining_tokens) / state.total_tokens) * 100
#     gauge_placeholder.markdown("🧮 Token Usage")
#     render_token_usage_gauge(current_percent)
#
#     batch_status.info(f"🔄 Processing: {state.current_chunk} of {state.batch_total}")
#     st.info(f"🪙 Remaining Tokens So Far: {state.remaining_tokens}")

