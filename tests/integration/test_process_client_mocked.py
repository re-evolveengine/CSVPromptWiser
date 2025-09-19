import json
import pandas as pd
import pytest
from unittest.mock import MagicMock

from model.core.chunk.chunker import DataFrameChunker
from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner
from model.io.result_saver import ResultSaver

@pytest.fixture
def dummy_df():
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
        'chunk_file': tmp_path / "chunks.json",
        'result_file': tmp_path / "results.json"
    }

def test_chunkwise_pause_resume_with_mock(dummy_df, prompt, tmp_paths):
    # === 1. Chunk and save ===
    chunker = DataFrameChunker(chunk_size=2)
    chunks = chunker.chunk_dataframe(dummy_df)
    DataFrameChunker.save_chunks_to_json(chunks, str(tmp_paths["chunk_file"]))

    # === 2. Setup ChunkManager ===
    manager = ChunkManager(str(tmp_paths["chunk_file"]))

    # === 3. Mock GeminiClient + ResilientRunner ===
    mock_client = MagicMock()
    mock_runner = GeminiResilientRunner(mock_client)

    mock_runner.run = MagicMock(side_effect=[
        "Chunk 1 summary",  # First call
        "Chunk 2 summary"   # Second call
    ])

    results = []

    def process_chunk(df: pd.DataFrame):
        response = mock_runner.run(prompt, df.to_dict(orient="records"))
        return {
            "chunk": df.to_dict(orient="records"),  # Needed for result saver
            "prompt": prompt,
            "response": response,
            "timestamp": "2025-07-24T12:34:56Z"
        }

    # === 4. Wrap processing function with pause logic ===
    call_count = [0]

    def wrapped_process(df):
        call_count[0] += 1
        if call_count[0] == 2:
            manager.pause()  # Pause after first chunk
        return process_chunk(df)

    manager.resume()
    results += manager.process_chunks(wrapped_process, show_progress=False)

    assert call_count[0] == 2
    assert manager.is_paused() is True

    # === 5. Resume and process remaining chunks ===
    manager.resume()
    results += manager.process_chunks(process_chunk, show_progress=False)

    # === 6. Save results ===
    ResultSaver.save_results_to_json(results, str(tmp_paths["result_file"]))

    # === 7. Validate output ===
    with open(tmp_paths["result_file"]) as f:
        saved = json.load(f)

    assert saved["version"] == 1.0
    assert len(saved["chunks"]) == manager.total_chunks
    assert all("response" in chunk for chunk in saved["chunks"])
    assert all("prompt" in chunk for chunk in saved["chunks"])
    assert all("data" in chunk for chunk in saved["chunks"])
