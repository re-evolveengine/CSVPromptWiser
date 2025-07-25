import json

import pytest
import pandas as pd
from pathlib import Path

from model.core.chunk_manager import ChunkManager


@pytest.fixture
def mock_json(tmp_path):
    """Create a mock JSON chunk file."""
    data = {
        "version": 1.0,
        "chunks": [
            {"chunk_id": "1", "data": [{"a": 1}, {"a": 2}]},
            {"chunk_id": "2", "data": [{"a": 3}, {"a": 4}]}
        ],
        "summary": {"total_chunks": 2, "processed_ids": []}
    }

    json_path = tmp_path / "test_chunks.json"
    json_path.write_text(json.dumps(data))
    return json_path


def test_initialization(mock_json):
    cm = ChunkManager(str(mock_json))
    assert cm.total_chunks == 2
    assert cm.remaining_chunks == 2
    assert not cm.is_paused()  # Should be running


def test_pause_resume_logic(mock_json):
    cm = ChunkManager(str(mock_json))
    assert cm.pause_event.is_set() is True

    cm.pause()
    assert cm.is_paused() is True

    cm.resume()
    assert cm.is_paused() is False


def test_get_next_chunk_and_mark_processed(mock_json):
    cm = ChunkManager(str(mock_json))

    chunk = cm.get_next_chunk()
    assert isinstance(chunk, pd.DataFrame)
    assert chunk.shape == (2, 1)
    assert cm._current_chunk_id == "1"

    cm.mark_chunk_processed()
    assert "1" in cm._processed_set


def test_process_chunks_calls_func_and_respects_pause(mocker, mock_json):
    cm = ChunkManager(str(mock_json))

    # Spy on the function being passed
    mock_func = mocker.Mock(side_effect=lambda df: f"sum={df['a'].sum()}")

    # Pause and resume to simulate the check
    cm.pause()
    assert cm.is_paused() is True
    cm.resume()

    results = cm.process_chunks(mock_func, show_progress=False)
    assert results == ["sum=3", "sum=7"]
    assert mock_func.call_count == 2


def test_process_chunks_handles_exceptions(mocker, mock_json):
    cm = ChunkManager(str(mock_json))

    def faulty_func(df):
        raise ValueError("boom")

    results = cm.process_chunks(faulty_func, show_progress=False)
    assert "boom" in results[0]
    assert "boom" in results[1]


def test_repr_output(mock_json):
    cm = ChunkManager(str(mock_json))
    rep = repr(cm)
    assert "processed=0/2" in rep
