# streamlit_dir/chunk_processor_panel.py

import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Any

from model.core.chunk.chunk_manager import ChunkManager
from model.io.gemini_result_saver import GeminiResultSaver
from model.utils.constants import RESULTS_DIR, TEMP_DIR
from streamlit_dir.ui.run_gemini_chunk_processor_ui import run_gemini_chunk_processor_ui


def process_chunks_ui(
        client: Any,
        prompt: str,
        chunk_file_path: str,
        max_chunks: int
):
    """
    Streamlit panel to process chunks with a pre-configured GeminiClient.

    Args:
        client: A GeminiClient (subclass of BaseLLMClient) already initialized
                with model name, API key, and generation_config.
        prompt: User’s prompt string.
        chunk_file_path: Path to the JSON file of chunks.
        max_chunks: Number of chunks to process.
    """
    st.markdown("### 🧠 Chunk Processing Progress")

    if not all([client, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    chunk_manager = ChunkManager(json_path=chunk_file_path)
    total = chunk_manager.total_chunks
    remaining = chunk_manager.remaining_chunks

    st.info(f"📦 Total Chunks: {total} 🔁 Remaining: {remaining}")

    # --- Progress indicators in main panel ---
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    # Always show the UI, but only process when start_processing is True
    if "start_processing" in st.session_state and st.session_state.get("start_processing"):
        try:
            # --- Run processor with callback ---
            results, errors = run_gemini_chunk_processor_ui(
                prompt=prompt,
                client=client,
                chunk_manager=chunk_manager,
                max_chunks=max_chunks,
                progress_callback=lambda i, n: [
                    progress_bar.progress(i / n),
                    status_placeholder.info(f"✅ Processed chunk {i} of {n}")
                ]
            )

            # --- Show any errors ---
            if errors:
                st.error("⚠️ Some chunks failed:")
                for err in errors:
                    st.write(f"- {err}")

            if not results:
                st.error("❌ No chunks were processed successfully.")
                return

            # --- Save and show result links ---
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
            # Reset the processing flag
            st.session_state["start_processing"] = False
            st.rerun()
    else:
        # Show idle state
        progress_bar.empty()
        status_placeholder.info("ℹ️ Click 'Start Processing' to begin chunk processing.")
