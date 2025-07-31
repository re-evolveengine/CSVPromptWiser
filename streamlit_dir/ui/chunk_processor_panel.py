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
        chunk_file_path: str = TEMP_DIR + "/chunks.json"
):
    """
    Streamlit panel to process chunks with a pre-configured GeminiClient.

    Args:
        client: A GeminiClient (subclass of BaseLLMClient) already initialized
                with model name, API key, and generation_config.
        prompt: User’s prompt string.
        chunk_file_path: Path to the JSON file of chunks.
    """
    st.markdown("### 🧠 Process Chunks with Gemini")

    if not all([client, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure client, prompt, and chunk file are all set.")
        return

    # Load chunk manager
    chunk_manager = ChunkManager(json_path=chunk_file_path)
    total = chunk_manager.total_chunks
    remaining = chunk_manager.remaining_chunks

    st.info(f"📦 Total Chunks: {total} 🔁 Remaining: {remaining}")

    # Let user pick how many to run
    num_chunks = st.number_input(
        "🔢 Number of chunks to process",
        min_value=1,
        max_value=remaining,
        value=min(remaining, 5),
        step=1
    )

    if st.button("🚀 Start Processing"):
        with st.spinner("Processing chunks…"):
            results, errors = run_gemini_chunk_processor_ui(
                prompt=prompt,
                client=client,
                chunk_manager=chunk_manager,
                max_chunks=int(num_chunks)
            )

        # Show any errors
        if errors:
            st.error("⚠️ Some chunks failed:")
            for err in errors:
                st.write(f"- {err}")

        if not results:
            st.error("❌ No chunks were processed successfully.")
            return

        # Save and show links
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{client.model}_{timestamp}"
        json_path = Path(RESULTS_DIR) / f"{base}.json"
        csv_path = Path(RESULTS_DIR) / f"{base}.csv"

        GeminiResultSaver.save_results_to_json(results, str(json_path))
        GeminiResultSaver.save_results_to_csv(results, str(csv_path))

        st.success("✅ Processing complete and results saved.")
        st.markdown(f"- 📁 [Download JSON Result]({json_path})")
        st.markdown(f"- 📁 [Download CSV Result]({csv_path})")
