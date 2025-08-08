import pytest
import pandas as pd
import json
from model.core.chunk.chunker import DataFrameChunker


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': range(1, 11),  # 10 rows
        'name': [f'Item {i}' for i in range(1, 11)],
        'value': [i * 10 for i in range(1, 11)]
    })


def test_chunk_dataframe_basic(sample_dataframe):
    """Test basic chunking functionality."""
    chunker = DataFrameChunker(chunk_size=3)
    chunks = chunker.chunk_dataframe(sample_dataframe)
    
    assert len(chunks) == 4  # 10 items / 3 per chunk = 4 chunks
    assert len(chunks[0]) == 3  # First chunk has 3 items
    assert len(chunks[-1]) == 1  # Last chunk has 1 item
    assert chunker.chunks == chunks  # Test property access


def test_chunk_dataframe_custom_size(sample_dataframe):
    """Test chunking with custom chunk size."""
    chunker = DataFrameChunker()
    chunks = chunker.chunk_dataframe(sample_dataframe, chunk_size=5)
    
    assert len(chunks) == 2  # 10 items / 5 per chunk = 2 chunks
    assert len(chunks[0]) == 5
    assert len(chunks[1]) == 5


def test_chunk_dataframe_empty():
    """Test chunking an empty DataFrame."""
    chunker = DataFrameChunker()
    df = pd.DataFrame()
    
    # Empty DataFrame returns an empty list of chunks
    chunks = chunker.chunk_dataframe(df)
    assert chunks == []
    
    # Verify save_chunks_to_json raises an error with empty chunks
    with pytest.raises(ValueError, match="No chunks to save"):
        chunker.save_chunks_to_json(chunks, "dummy.json")


def test_chunk_dataframe_invalid_size(sample_dataframe):
    """Test chunking with invalid chunk size uses default chunk size."""
    chunker = DataFrameChunker(chunk_size=1000)  # Explicit default size
    
    # Test that invalid sizes use the default chunk size
    chunks = chunker.chunk_dataframe(sample_dataframe, chunk_size=0)
    assert len(chunks) == 1  # All rows in one chunk with default size 1000
    
    chunks = chunker.chunk_dataframe(sample_dataframe, chunk_size=-5)
    assert len(chunks) == 1  # All rows in one chunk with default size 1000


def test_save_chunks_to_json(sample_dataframe, tmp_path):
    """Test saving chunks to JSON file."""
    # Setup
    chunker = DataFrameChunker(chunk_size=4)
    chunks = chunker.chunk_dataframe(sample_dataframe)
    output_file = tmp_path / "output.json"
    
    # Test with metadata and row limit
    metadata = {"source": "test", "version": 1}
    chunker.save_chunks_to_json(
        chunks,
        str(output_file),
        max_rows_per_chunk=2,
        metadata=metadata
    )
    
    # Verify file was created
    assert output_file.exists()
    
    # Verify content
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Check metadata and version
    assert data['version'] == 1.0
    assert data['metadata'] == metadata
    
    # Check chunks structure
    assert len(data['chunks']) == 3  # 3 chunks (4,4,2) with max 2 rows per chunk
    assert data['summary']['total_chunks'] == 3
    assert 'processed_ids' in data['summary']
    
    # Check first chunk data
    first_chunk = data['chunks'][0]
    assert 'chunk_id' in first_chunk  # Now using UUID instead of chunk_number
    assert len(first_chunk['data']) == 2  # Limited by max_rows_per_chunk
    assert first_chunk['original_rows'] == 4  # Original chunk size


def test_chunks_property_before_chunking():
    """Test accessing chunks property before chunking raises error."""
    chunker = DataFrameChunker()
    
    # The implementation raises a ValueError with a specific message
    with pytest.raises(ValueError, match="No chunks available - run chunk_dataframe\(\) first"):
        _ = chunker.chunks


def test_chunk_dataframe_adds_source_id(sample_dataframe):
    """Test that chunk_dataframe adds a source_id column with UUIDs."""
    chunker = DataFrameChunker(chunk_size=3)
    chunks = chunker.chunk_dataframe(sample_dataframe)
    
    # Check that source_id was added to each chunk
    for chunk in chunks:
        assert 'source_id' in chunk.columns
        # Verify all source_ids are strings that look like UUIDs
        assert all(isinstance(id, str) and len(id) == 36 for id in chunk['source_id'])
        # Verify all source_ids are unique within the chunk
        assert len(chunk['source_id']) == len(chunk['source_id'].unique())


def test_save_chunks_preserves_source_id(sample_dataframe, tmp_path):
    """Test that source_id is preserved when saving chunks to JSON."""
    chunker = DataFrameChunker(chunk_size=4)
    chunks = chunker.chunk_dataframe(sample_dataframe)
    output_file = tmp_path / "output.json"
    
    chunker.save_chunks_to_json(chunks, str(output_file))
    
    # Verify file was created
    assert output_file.exists()
    
    # Verify content
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Check that source_id is present in the saved data
    for chunk in data['chunks']:
        for row in chunk['data']:
            assert 'source_id' in row
            assert isinstance(row['source_id'], str)
            assert len(row['source_id']) == 36  # UUID string length
