import json
import pandas as pd
import pytest
from unittest.mock import MagicMock

from model.core.chunk.chunker import DataFrameChunker
from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner
from model.io.gemini_result_saver import GeminiResultSaver

@pytest.fixture
def faulty_df():
    return pd.DataFrame({
        'product': ['A', 'B', 'C', 'D'],
        'sales': [100, 200, 300, 400]
    })

@pytest.fixture
def prompt():
    return "Summarize the sales performance in this table:"

@pytest.fixture
def tmp_paths(tmp_path):
    return {
        'chunk_file': tmp_path / "faulty_chunks.json",
        'result_file': tmp_path / "faulty_results.json"
    }

def test_chunkwise_with_error_handling(faulty_df, prompt, tmp_paths):
    # === 1. Chunk and save ===
    chunker = DataFrameChunker(chunk_size=2)
    chunks = chunker.chunk_dataframe(faulty_df)
    DataFrameChunker.save_chunks_to_json(chunks, str(tmp_paths["chunk_file"]))

    # === 2. Load ChunkManager ===
    manager = ChunkManager(str(tmp_paths["chunk_file"]))

    # === 3. Mock GeminiClient + ResilientRunner ===
    mock_client = MagicMock()
    mock_runner = GeminiResilientRunner(mock_client)

    # First chunk will raise an error, second will succeed
    mock_runner.run = MagicMock(side_effect=[
        Exception("Simulated failure for chunk 1"),
        "Chunk 2 summary"
    ])

    results = []

    def process_chunk(df: pd.DataFrame):
        try:
            response = mock_runner.run(prompt, df.to_dict(orient="records"))
            return {
                "chunk": df,
                "prompt": prompt,
                "response": response,
                "timestamp": "2025-07-24T12:34:56Z"
            }
        except Exception as e:
            return {
                "chunk": df,
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "timestamp": "2025-07-24T12:34:56Z"
            }

    manager.resume()
    results += manager.process_chunks(process_chunk, show_progress=False)

    # === 4. Save results to file ===
    GeminiResultSaver.save_results_to_json(results, str(tmp_paths["result_file"]))

    # === 5. Validate output ===
    with open(tmp_paths["result_file"]) as f:
        saved = json.load(f)

    assert saved["version"] == 1.0
    assert len(saved["chunks"]) == manager.total_chunks
    assert any("Error:" in c["response"] for c in saved["chunks"])
    assert all("prompt" in c for c in saved["chunks"])
    assert all("data" in c for c in saved["chunks"])  

    # === 6. Optionally validate state ===
    processed_ids = saved.get("summary", {}).get("processed_ids", [])
    assert len(processed_ids) == 2  

    # === 7. Ensure chunk with error was NOT marked as processed
    assert manager.remaining_chunks == 0  
