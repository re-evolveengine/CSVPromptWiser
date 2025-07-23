import pytest
import pandas as pd
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from model.core.chunk_manager import ChunkManager


@pytest.fixture
def sample_chunk_data():
    """Create a sample chunked JSON file for testing."""
    return {
        "metadata": {"source": "test"},
        "chunks": [
            {"chunk_number": 1, "data": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]},
            {"chunk_number": 2, "data": [{"id": 3, "name": "Item 3"}]}
        ],
        "summary": {"total_chunks": 2, "saved_rows": 3}
    }


@pytest.fixture
def sample_chunk_file(tmp_path, sample_chunk_data):
    """Create a temporary JSON file with sample chunk data."""
    file_path = tmp_path / "test_chunks.json"
    with open(file_path, 'w') as f:
        json.dump(sample_chunk_data, f)
    return file_path


def test_chunk_manager_init_valid(sample_chunk_file):
    """Test initialization with a valid JSON file."""
    manager = ChunkManager(str(sample_chunk_file))
    assert manager.total_chunks == 2
    assert manager.remaining_chunks == 2


def test_chunk_manager_init_invalid_file(tmp_path):
    """Test initialization with invalid file paths."""
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        ChunkManager("nonexistent.json")
    
    # Non-JSON file
    txt_file = tmp_path / "test.txt"
    txt_file.touch()
    with pytest.raises(ValueError, match="must be JSON format"):
        ChunkManager(str(txt_file))


def test_get_next_chunk(sample_chunk_file):
    """Test retrieving chunks sequentially."""
    manager = ChunkManager(str(sample_chunk_file))
    
    # First chunk
    chunk1 = manager.get_next_chunk()
    assert isinstance(chunk1, pd.DataFrame)
    assert len(chunk1) == 2
    assert manager.remaining_chunks == 1
    
    # Second chunk
    chunk2 = manager.get_next_chunk()
    assert len(chunk2) == 1
    assert manager.remaining_chunks == 0
    
    # No more chunks
    assert manager.get_next_chunk() is None
    assert manager.remaining_chunks == 0


def test_process_chunks(sample_chunk_file):
    """Test processing chunks with a function."""
    manager = ChunkManager(str(sample_chunk_file))
    
    # Simple processing function
    def process_chunk(df):
        return len(df)
    
    results = manager.process_chunks(process_chunk, show_progress=False)
    assert results == [2, 1]  # Lengths of our test chunks
    assert manager.remaining_chunks == 0


@patch('tqdm.tqdm')
def test_process_chunks_with_progress(mock_tqdm, sample_chunk_file):
    """Test progress bar is shown when requested."""
    manager = ChunkManager(str(sample_chunk_file))
    
    # Mock tqdm iterator
    mock_iterator = MagicMock()
    mock_iterator.__iter__.return_value = range(2)
    mock_tqdm.return_value = mock_iterator
    
    manager.process_chunks(lambda x: x, show_progress=True)
    
    # Verify tqdm was called with correct parameters
    mock_tqdm.assert_called_once()
    assert "test_chunks.json" in mock_tqdm.call_args[1]["desc"]
    assert mock_tqdm.call_args[1]["unit"] == "chunk"


def test_repr(sample_chunk_file):
    """Test string representation."""
    manager = ChunkManager(str(sample_chunk_file))
    assert "ChunkManager" in repr(manager)
    assert "test_chunks.json" in repr(manager)
    assert "0/2" in repr(manager)  # Initial state
    
    # After processing one chunk
    manager.get_next_chunk()
    assert "1/2" in repr(manager)
