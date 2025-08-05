import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Any

from model.core.chunk.chunk_manager import ChunkManager
from model.io.gemini_result_saver import GeminiResultSaver
from model.io.model_prefs import ModelPreference
from model.utils.constants import RESULTS_DIR
from streamlit_dir.chunk_process_state import ChunkProcessorState
from streamlit_dir.ui.run_gemini_chunk_processor_ui import run_gemini_chunk_processor_ui
from streamlit_dir.ui.token_usage_gauge import render_token_usage_gauge


def process_chunks_ui(
        client: Any,
        prompt: str,
        chunk_file_path: str,
        max_chunks: int,
        total_tokens: int
):
    """
    Streamlit panel to process chunks with a pre-configured GeminiClient.
    """

    st.markdown("### 🧠 Chunk Processing Progress")

    if not all([client, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    # Init state object if not present
    if "chunk_state" not in st.session_state:
        st.session_state.chunk_state = ChunkProcessorState(total_tokens=total_tokens)

    state: ChunkProcessorState = st.session_state.chunk_state

    chunk_manager = ChunkManager(json_path=chunk_file_path)
    total_chunks = chunk_manager.total_chunks
    remaining_chunks = chunk_manager.remaining_chunks

    # UI placeholders
    status_container = st.empty()
    gauge_placeholder = st.empty()
    progress_bar = st.progress(0)
    batch_status = st.empty()
    status_placeholder = st.empty()

    def update_status(current: int, batch_total: int, remaining_tokens: int):
        """Callback to update processing state only."""
        state.current_chunk = current
        state.batch_total = batch_total
        state.remaining_tokens = remaining_tokens

    # ⏳ Display Initial Status
    status_container.info(f"📦 Total Chunks: {total_chunks} 🔁 Remaining: {remaining_chunks}")
    batch_status.info("⏳ Waiting to start processing...")


    # 🔄 Processing
    if "start_processing" in st.session_state and st.session_state.get("start_processing"):
        try:
            results, errors = run_gemini_chunk_processor_ui(
                prompt=prompt,
                client=client,
                chunk_manager=chunk_manager,
                max_chunks=max_chunks,
                progress_callback=update_status
            )

            if errors:
                st.error("⚠️ Some chunks failed:")
                for err in errors:
                    st.write(f"- {err}")

            if not results:
                st.error("❌ No chunks were processed successfully.")
                return

            # ✅ Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{client.model}_{timestamp}"
            json_path = Path(RESULTS_DIR) / f"{base}.json"
            csv_path = Path(RESULTS_DIR) / f"{base}.csv"

            GeminiResultSaver.save_results_to_json(results, str(json_path))
            GeminiResultSaver.save_results_to_csv(results, str(csv_path))

            st.success("✅ Processing complete and results saved.")
            # Optionally: Show download links
            # st.markdown(f"- 📁 [Download JSON Result]({json_path})")
            # st.markdown(f"- 📁 [Download CSV Result]({csv_path})")

        finally:
            st.session_state["start_processing"] = False
            st.rerun()

    else:
        # Idle state
        progress_bar.empty()
        status_placeholder.info("ℹ️ Click 'Start Processing' to begin chunk processing.")


    # 🎯 Show token usage gauge
    st.subheader("🧮 Token Usage Summary")
    current_percent = ((state.total_tokens - state.remaining_tokens) / state.total_tokens) * 100
    gauge_placeholder.markdown("🧮 Token Usage")
    render_token_usage_gauge(current_percent)

    batch_status.info(f"🔄 Processing: {state.current_chunk} of {state.batch_total}")
    st.info(f"🪙 Remaining Tokens So Far: {state.remaining_tokens}")

