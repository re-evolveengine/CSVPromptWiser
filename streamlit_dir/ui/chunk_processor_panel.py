# streamlit_dir/chunk_processor_panel.py

import streamlit as st
from datetime import datetime
from pathlib import Path

from model.core.chunk.chunk_manager import ChunkManager
from model.io.gemini_result_saver import GeminiResultSaver
from model.utils.constants import RESULTS_DIR, TEMP_DIR

from cli.cli_utils import run_gemini_chunk_processor


def process_chunks_ui(api_key: str, model_name: str, prompt: str, chunk_file_path: str):
    st.markdown("### 🧠 Process Chunks with Gemini")

    if not all([api_key, model_name, prompt, chunk_file_path]):
        st.warning("⚠️ Please make sure model, API key, prompt, and chunk file are all set.")
        return

    chunk_manager = ChunkManager(json_path=chunk_file_path)
    total_chunks = chunk_manager.total_chunks
    remaining_chunks = chunk_manager.remaining_chunks

    st.info(f"📦 Total Chunks: {total_chunks} | 🔁 Remaining: {remaining_chunks}")

    num_chunks = st.number_input("🔢 Number of chunks to process", min_value=1, max_value=remaining_chunks, value=min(remaining_chunks, 5), step=1)

    if st.button("🚀 Start Processing"):
        with st.spinner("Processing chunks..."):
            results, success = run_gemini_chunk_processor(
                prompt=prompt,
                model_name=model_name,
                api_key=api_key,
                chunk_manager=chunk_manager,
                max_chunks=num_chunks,
                show_progress=True
            )

        if not success or not results:
            st.error("❌ No chunks processed successfully.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name}_{timestamp}"
        json_path = Path(RESULTS_DIR) / f"{base_name}.json"
        csv_path = Path(RESULTS_DIR) / f"{base_name}.csv"

        GeminiResultSaver.save_results_to_json(results, str(json_path))
        GeminiResultSaver.save_results_to_csv(results, str(csv_path))

        st.success("✅ Processing complete and results saved.")
        st.markdown(f"- 📁 [JSON Result]({json_path})")
        st.markdown(f"- 📁 [CSV Result]({csv_path})")
