"""Integration tests for the Gemini client with chunking functionality."""
import pytest
from unittest.mock import patch, MagicMock

from model.core.chunk.chunker import DataFrameChunker
from model.core.chunk.chunk_manager import ChunkManager
from cli import DatasetLoader

# Sample data for testing
SAMPLE_CSV = """id,name,value
1,Item 1,10
2,Item 2,20
3,Item 3,30
4,Item 4,40
5,Item 5,50
"""

@pytest.fixture(scope="module")
def sample_csv_file(tmp_path_factory):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path_factory.mktemp("data") / "test_data.csv"
    csv_file.write_text(SAMPLE_CSV)
    return str(csv_file)

@pytest.fixture
def mock_gemini_response():
    """Mock response from Gemini API."""
    mock_response = MagicMock()
    mock_response.text = "Mocked Gemini response"
    return mock_response

def test_gemini_integration_with_chunking(sample_csv_file, mock_gemini_response, tmp_path, monkeypatch):
    """Test Gemini client integration with chunking functionality."""
    # Setup
    max_chunk_size = 2  # As per requirement
    json_path = tmp_path / "chunks.json"
    
    # Mock the Gemini API call
    with patch('google.generativeai.GenerativeModel') as mock_model:
        # Configure the mock
        mock_instance = mock_model.return_value
        mock_instance.generate_content.return_value = mock_gemini_response
        
        # Initialize components
        loader = DatasetLoader()
        df = loader.load_from_upload(sample_csv_file)
        
        # Test chunking with max_chunk_size
        chunker = DataFrameChunker(chunk_size=max_chunk_size)
        chunks = chunker.chunk_dataframe(df)
        
        # Verify chunk sizes
        assert len(chunks) == 3  # 5 rows with chunk size 2 should give 3 chunks
        assert len(chunks[0]) == max_chunk_size
        assert len(chunks[1]) == max_chunk_size
        assert len(chunks[2]) == 1  # Last chunk with remaining row
        
        # Save chunks to JSON
        chunker.save_chunks_to_json(chunks=chunks, file_path=str(json_path))
        
        # Initialize ChunkManager with the saved chunks
        manager = ChunkManager(str(json_path))
        
        # Mock the process_chunks function
        def mock_process_chunk(chunk):
            # Verify chunk size doesn't exceed max_chunk_size
            assert len(chunk) <= max_chunk_size
            return f"Processed {len(chunk)} rows"
        
        # Process chunks one by one and mark them as processed
        results = []
        for _ in range(3):  # We know we have 3 chunks
            chunk = manager.get_next_chunk()
            assert chunk is not None
            result = mock_process_chunk(chunk)
            results.append(result)
            manager.mark_chunk_processed()
        
        # Verify results
        assert len(results) == 3  # Should process all 3 chunks
        assert all(isinstance(result, str) for result in results)
        
        # Verify all chunks were processed
        assert manager.remaining_chunks == 0, f"Expected 0 remaining chunks, but got {manager.remaining_chunks}"
