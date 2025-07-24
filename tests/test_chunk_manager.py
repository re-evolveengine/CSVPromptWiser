import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
from model.core.chunk_manager import ChunkManager


@pytest.fixture
def sample_chunk_data():
    """Create sample chunk data for testing."""
    return {
        "metadata": {"source": "test", "version": 1},
        "summary": {"total_chunks": 3, "processed_chunks": 0},
        "chunks": [
            {"chunk_number": 1, "original_rows": 3, "saved_rows": 3, 
             "data": [{"id": 1, "name": "Item 1", "value": 10},
                     {"id": 2, "name": "Item 2", "value": 20},
                     {"id": 3, "name": "Item 3", "value": 30}]},
            {"chunk_number": 2, "original_rows": 3, "saved_rows": 3,
             "data": [{"id": 4, "name": "Item 4", "value": 40},
                     {"id": 5, "name": "Item 5", "value": 50},
                     {"id": 6, "name": "Item 6", "value": 60}]},
            {"chunk_number": 3, "original_rows": 2, "saved_rows": 2,
             "data": [{"id": 7, "name": "Item 7", "value": 70},
                     {"id": 8, "name": "Item 8", "value": 80}]}
        ]
    }


@pytest.fixture
def chunk_file(tmp_path, sample_chunk_data):
    """Create a temporary chunk file for testing."""
    file_path = tmp_path / "test_chunks.json"
    with open(file_path, 'w') as f:
        json.dump(sample_chunk_data, f)
    return file_path


def test_chunk_manager_initialization(chunk_file):
    """Test ChunkManager initialization and properties."""
    manager = ChunkManager(str(chunk_file))
    
    assert manager.total_chunks == 3
    assert manager.remaining_chunks == 3
    assert manager._current_chunk == 0
    assert manager._processed_chunks == 0


def test_chunk_manager_invalid_file(tmp_path):
    """Test initialization with invalid file paths."""
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        ChunkManager(str(tmp_path / "nonexistent.json"))
    
    # Test non-JSON file
    txt_file = tmp_path / "test.txt"
    txt_file.touch()
    with pytest.raises(ValueError, match="File must be JSON format"):
        ChunkManager(str(txt_file))


def test_get_next_chunk(chunk_file, sample_chunk_data):
    """Test retrieving chunks sequentially."""
    manager = ChunkManager(str(chunk_file))
    
    # Get first chunk
    chunk1 = manager.get_next_chunk()
    assert isinstance(chunk1, pd.DataFrame)
    assert len(chunk1) == 3
    assert chunk1['id'].tolist() == [1, 2, 3]
    assert manager.remaining_chunks == 2
    
    # Get second chunk
    chunk2 = manager.get_next_chunk()
    assert len(chunk2) == 3
    assert chunk2['id'].tolist() == [4, 5, 6]
    assert manager.remaining_chunks == 1
    
    # Get third chunk
    chunk3 = manager.get_next_chunk()
    assert len(chunk3) == 2
    assert chunk3['id'].tolist() == [7, 8]
    assert manager.remaining_chunks == 0
    
    # No more chunks
    assert manager.get_next_chunk() is None


def test_mark_chunk_processed(chunk_file, sample_chunk_data):
    """Test marking chunks as processed."""
    # Create a fresh copy of the file for this test
    test_file = chunk_file.parent / "test_mark_processed.json"
    with open(test_file, 'w') as f:
        json.dump(sample_chunk_data, f)
    
    manager = ChunkManager(str(test_file))
    
    # Mark first chunk as processed
    manager.mark_chunk_processed(0)
    
    # Verify the chunk was removed from the file
    with open(test_file, 'r') as f:
        updated_data = json.load(f)
    
    assert len(updated_data['chunks']) == 2  # One less chunk
    assert updated_data['summary']['processed_chunks'] == 1
    assert updated_data['chunks'][0]['chunk_number'] == 2  # Second chunk is now first


def test_mark_chunk_processed_invalid_index(chunk_file):
    """Test marking invalid chunk index as processed."""
    manager = ChunkManager(str(chunk_file))
    
    with pytest.raises(IndexError):
        manager.mark_chunk_processed(-1)  # Negative index
    
    with pytest.raises(IndexError):
        manager.mark_chunk_processed(10)  # Out of bounds index


def test_process_chunks(chunk_file):
    """Test processing chunks with a function."""
    manager = ChunkManager(str(chunk_file))
    
    # Define a simple processing function
    def process_func(df):
        return {
            'count': len(df),
            'sum': df['value'].sum(),
            'ids': df['id'].tolist()
        }
    
    # Process chunks
    results = manager.process_chunks(process_func, show_progress=False)
    
    # Verify results
    assert len(results) == 3
    assert results[0] == {'count': 3, 'sum': 60, 'ids': [1, 2, 3]}
    assert results[1] == {'count': 3, 'sum': 150, 'ids': [4, 5, 6]}
    assert results[2] == {'count': 2, 'sum': 150, 'ids': [7, 8]}
    assert manager.remaining_chunks == 0


def test_repr(chunk_file):
    """Test string representation of ChunkManager."""
    manager = ChunkManager(str(chunk_file))
    assert str(manager) == f"ChunkManager(file='test_chunks.json', processed=0/3)"
    
    # Process a chunk and check updated representation
    manager.get_next_chunk()
    assert str(manager) == f"ChunkManager(file='test_chunks.json', processed=1/3)"
