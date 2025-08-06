import json
import pytest
import pandas as pd

from model.core.chunk.chunk_manager import ChunkManager
from model.utils.constants import JSON_CHUNK_VERSION


@pytest.fixture
def sample_chunk_data():
    """Create sample chunk data for testing."""
    return {
        "version": JSON_CHUNK_VERSION,
        "metadata": {"source": "test", "version": 1},
        "chunks": [
            {
                "chunk_id": "chunk1",
                "data": [
                    {"id": 1, "name": "Item 1", "value": 10},
                    {"id": 2, "name": "Item 2", "value": 20}
                ],
                "original_rows": 2
            },
            {
                "chunk_id": "chunk2",
                "data": [
                    {"id": 3, "name": "Item 3", "value": 30},
                    {"id": 4, "name": "Item 4", "value": 40}
                ],
                "original_rows": 2
            }
        ],
        "summary": {
            "total_chunks": 2,
            "processed_ids": []
        }
    }


@pytest.fixture
def chunk_manager(tmp_path, sample_chunk_data):
    """Create a ChunkManager instance with a temporary JSON file."""
    temp_file = tmp_path / "test_chunks.json"
    with open(temp_file, 'w') as f:
        json.dump(sample_chunk_data, f)
    return ChunkManager(str(temp_file))


def test_chunk_manager_initialization(chunk_manager, sample_chunk_data):
    """Test ChunkManager initialization and properties."""
    assert chunk_manager.total_chunks == 2
    assert chunk_manager.remaining_chunks == 2
    assert not chunk_manager.is_paused()


def test_get_next_chunk(chunk_manager):
    """Test getting the next unprocessed chunk."""
    # First call should return the first chunk
    df = chunk_manager.get_next_chunk()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.iloc[0]['id'] == 1
    assert df.iloc[1]['id'] == 2

    # Mark first chunk as processed
    chunk_manager.mark_chunk_processed()

    # Next call should return the second chunk
    df = chunk_manager.get_next_chunk()
    assert len(df) == 2
    assert df.iloc[0]['id'] == 3
    assert df.iloc[1]['id'] == 4

    # Mark second chunk as processed
    chunk_manager.mark_chunk_processed()

    # No more chunks should return None
    assert chunk_manager.get_next_chunk() is None


def test_mark_chunk_processed(chunk_manager):
    """Test marking chunks as processed."""
    assert chunk_manager.remaining_chunks == 2

    # Get and mark first chunk as processed
    df = chunk_manager.get_next_chunk()
    chunk_manager.mark_chunk_processed()

    assert chunk_manager.remaining_chunks == 1
    assert "chunk1" in chunk_manager._processed_set


def test_process_chunks(chunk_manager):
    """Test processing chunks with a function."""
    # Define a simple processing function
    def process_func(df):
        return {"count": len(df), "sum": df['value'].sum()}

    # Process all chunks
    results = chunk_manager.process_chunks(process_func)

    # Verify results
    assert len(results) == 2
    assert results[0] == {"count": 2, "sum": 30}  # 10 + 20
    assert results[1] == {"count": 2, "sum": 70}  # 30 + 40

    # Verify all chunks are marked as processed
    assert chunk_manager.remaining_chunks == 0


def test_process_chunks_with_max(chunk_manager):
    """Test processing a limited number of chunks."""
    def process_func(df):
        return len(df)

    # Process only 1 chunk
    results = chunk_manager.process_chunks(process_func, chunk_count=1)

    assert len(results) == 1
    assert results[0] == 2  # 2 rows in the first chunk
    assert chunk_manager.remaining_chunks == 1  # 1 chunk remaining


def test_pause_resume(chunk_manager):
    """Test pausing and resuming chunk processing."""
    chunk_manager.pause()
    assert chunk_manager.is_paused()

    chunk_manager.resume()
    assert not chunk_manager.is_paused()


def test_invalid_json_version(tmp_path):
    """Test initialization with invalid JSON version."""
    invalid_data = {"version": 0.9, "chunks": [], "summary": {"total_chunks": 0}}
    temp_file = tmp_path / "invalid.json"
    with open(temp_file, 'w') as f:
        json.dump(invalid_data, f)

    with pytest.raises(ValueError, match="Unsupported or missing JSON version"):
        ChunkManager(str(temp_file))