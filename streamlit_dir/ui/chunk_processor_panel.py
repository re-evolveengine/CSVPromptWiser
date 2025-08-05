# streamlit_dir/chunk_processor_panel.py

import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Any

from model.core.chunk.chunk_manager import ChunkManager
from model.io.gemini_result_saver import GeminiResultSaver
from model.io.model_prefs import ModelPreference
from model.utils.constants import RESULTS_DIR, TEMP_DIR
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

    Args:
        client: A GeminiClient (subclass of BaseLLMClient) already initialized
                with model name, API key, and generation_config.
        prompt: User’s prompt string.
        chunk_file_path: Path to the JSON file of chunks.
        max_chunks: Number of chunks to process.
        total_tokens: Initial total token quota to track remaining tokens.
    """
    st.markdown("### 🧠 Chunk Processing Progress")

    if not all([client, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    chunk_manager = ChunkManager(json_path=chunk_file_path)
    total = chunk_manager.total_chunks
    remaining = chunk_manager.remaining_chunks

    # Preferences and token gauge
    prefs = ModelPreference()

    # UI placeholders
    status_container = st.empty()
    gauge_placeholder = st.empty()
    progress_bar = st.progress(0)
    batch_status = st.empty()
    status_placeholder = st.empty()

    def update_token_gauge(percent: float):
        """Update the token usage gauge display."""
        gauge_placeholder.empty()
        gauge_placeholder.markdown("🧮 Token Usage Summary")
        render_token_usage_gauge(percent)

    def update_status(current: int, batch_total: int, remaining_tokens: int = 0):
        current_remaining = chunk_manager.remaining_chunks
        percent_used = ((total_tokens - remaining_tokens) / total_tokens) * 100 if total_tokens > 0 else 0

        # Show info
        status_container.info(f"📦 Total Chunks: {total} 🔁 Remaining: {current_remaining}")
        st.info(f"🪙 Remaining Tokens So Far: {remaining_tokens}")
        batch_status.info(f"🔄 Processing: {current} of {batch_total} in current batch")

        # Update gauge
        update_token_gauge(percent_used)

        # Update progress bar
        if batch_total > 0:
            progress_bar.progress(current / batch_total)

    # --- Initial render before processing starts ---
    st.subheader("🧮 Token Usage Summary")
    initial_remaining = prefs.get_remaining_total_tokens()
    initial_percent = ((total_tokens - initial_remaining) / total_tokens) * 100 if total_tokens > 0 else 0
    update_token_gauge(initial_percent)

    # Initial status display
    status_container.info(f"📦 Total Chunks: {total} 🔁 Remaining: {remaining}")
    batch_status.info("⏳ Waiting to start processing...")

    # --- Main processing ---
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

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{client.model}_{timestamp}"
            json_path = Path(RESULTS_DIR) / f"{base}.json"
            csv_path = Path(RESULTS_DIR) / f"{base}.csv"

            GeminiResultSaver.save_results_to_json(results, str(json_path))
            GeminiResultSaver.save_results_to_csv(results, str(csv_path))

            st.success("✅ Processing complete and results saved.")
            # st.markdown(f"- 📁 [Download JSON Result]({json_path})")
            # st.markdown(f"- 📁 [Download CSV Result]({csv_path})")

        finally:
            st.session_state["start_processing"] = False
            st.rerun()
    else:
        # Show idle state
        progress_bar.empty()
        status_placeholder.info("ℹ️ Click 'Start Processing' to begin chunk processing.")
