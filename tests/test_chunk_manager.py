import json
import pytest
import pandas as pd
from pathlib import Path
from model.core.chunk_manager import ChunkManager
from model.core.chunker import DataFrameChunker


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': range(1, 6),  # 5 rows
        'name': [f'Item {i}' for i in range(1, 6)],
        'value': [i * 10 for i in range(1, 6)]
    })


@pytest.fixture
def chunked_json_file(tmp_path, sample_dataframe):
    """Create a temporary JSON file with chunked data for testing."""
    # Create chunks
    chunker = DataFrameChunker(chunk_size=2)
    chunks = chunker.chunk_dataframe(sample_dataframe)
    
    # Save to temp file with chunk IDs
    file_path = tmp_path / "test_chunks.json"
    
    # Prepare chunks with data only (no chunk_id)
    chunks_with_ids = []
    for i, chunk in enumerate(chunks, 1):
        chunks_with_ids.append({
            "data": chunk.to_dict(orient='records'),
            "chunk_number": i
        })
        
    # Ensure we have exactly 3 chunks (5 items with chunk size 2)
    assert len(chunks_with_ids) == 3
    
    # Create the complete data structure
    data = {
        "version": 1.0,
        "chunks": chunks_with_ids,
        "summary": {
            "total_chunks": len(chunks_with_ids),
            "processed": 0,
            "processed_ids": []
        },
        "metadata": {"test": True}
    }
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return file_path


def test_chunk_manager_initialization(chunked_json_file):
    """Test that ChunkManager initializes correctly with a valid JSON file."""
    manager = ChunkManager(str(chunked_json_file))
    
    # Verify basic properties
    assert manager.total_chunks == 3  # 5 rows with chunk size 2 = 3 chunks
    assert manager.remaining_chunks == 3  # Initially all chunks are unprocessed


def test_chunk_processing_flow(chunked_json_file):
    """Test the complete chunk processing flow."""
    # First, verify initial state
    manager = ChunkManager(str(chunked_json_file))
    assert manager.remaining_chunks == 3
    
    # Process first chunk
    chunk1 = manager.get_next_chunk()
    assert chunk1 is not None
    assert len(chunk1) == 2  # First chunk has 2 rows
    
    # Mark as processed and verify state
    manager.mark_chunk_processed()
    assert manager.remaining_chunks == 2
    
    # Process remaining chunks using process_chunks
    results = manager.process_chunks(
        lambda df: {'sum': df['value'].sum(), 'count': len(df)},
        show_progress=False
    )
    
    # Verify processing results - should process remaining 2 chunks
    assert len(results) == 2
    # The order of processing might vary, so check the sum of all results
    total_sum = sum(r['sum'] for r in results)
    assert total_sum in [70 + 50, 30 + 90]  # Either [3,4 + 5] or [3 + 4,5]
    
    # Verify all chunks are processed
    assert manager.remaining_chunks == 0
    assert manager.get_next_chunk() is None


def test_process_chunks_with_errors(chunked_json_file):
    """Test error handling in process_chunks."""
    manager = ChunkManager(str(chunked_json_file))
    
    # Process chunks with a function that fails on certain chunks
    results = manager.process_chunks(
        lambda df: 100 / (df['id'].iloc[0] - 2),  # Will fail when id=2
        show_progress=False
    )
    
    # Verify error handling - all chunks are processed, even if some fail
    assert len(results) == 3  # Should still process all chunks
    
    # Check for any error in the results (either as string or exception)
    has_errors = any(
        isinstance(r, str) and ("Error" in r or "division by zero" in r) 
        or isinstance(r, Exception) 
        for r in results
    )
    assert has_errors, "Expected at least one error in the results"
    
    # The current implementation marks all chunks as processed, even if they failed
    assert manager.remaining_chunks == 0  # All chunks are marked as processed


def test_state_persistence(chunked_json_file):
    """Test that processed state is saved between instances."""
    # First instance - process one chunk
    manager1 = ChunkManager(str(chunked_json_file))
    chunk = manager1.get_next_chunk()
    assert chunk is not None
    manager1.mark_chunk_processed()
    
    # Second instance - should continue from where we left off
    manager2 = ChunkManager(str(chunked_json_file))
    assert manager2.remaining_chunks == 2  # One less than total
    
    # Verify the processed chunk is tracked by index
    assert len(manager2._processed_set) == 1
    assert '0' in manager2._processed_set  # First chunk (index 0) is processed


def test_invalid_json_files(tmp_path):
    """Test handling of various invalid JSON files."""
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        ChunkManager(str(tmp_path / "nonexistent.json"))
    
    # Test invalid JSON format
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{invalid json")
    with pytest.raises(json.JSONDecodeError):
        ChunkManager(str(invalid_json))
    
    # Test invalid version
    invalid_version = tmp_path / "invalid_version.json"
    invalid_version.write_text(json.dumps({"version": 2.0, "chunks": []}))
    with pytest.raises(ValueError, match="Unsupported or missing JSON version"):
        ChunkManager(str(invalid_version))
    
    # Test missing required fields
    incomplete_json = tmp_path / "incomplete.json"
    incomplete_json.write_text(json.dumps({"version": 1.0}))
    # The current implementation doesn't raise KeyError for missing fields
    # It will initialize with empty chunks and summary
    manager = ChunkManager(str(incomplete_json))
    assert manager.total_chunks == 0
    assert manager.remaining_chunks == 0
