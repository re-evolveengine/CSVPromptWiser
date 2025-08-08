import json
import sys
import pytest
from pathlib import Path
import pandas as pd

from model.core.chunk.chunk_manager import ChunkManager
from utils import JSON_CHUNK_VERSION

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Import using relative path


def create_test_json_file(tmp_path, chunks_data=None, processed_ids=None):    
    if chunks_data is None:
        chunks_data = [
            {"chunk_id": 1, "data": [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]},
            {"chunk_id": 2, "data": [{"col1": 3, "col2": "c"}, {"col1": 4, "col2": "d"}]},
        ]
    
    if processed_ids is None:
        processed_ids = []
    
    data = {
        "version": JSON_CHUNK_VERSION,
        "chunks": chunks_data,
        "summary": {
            "total_chunks": len(chunks_data),
            "processed": len(processed_ids),
            "processed_ids": processed_ids
        }
    }
    
    file_path = tmp_path / "test_chunks.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    return file_path


class TestChunkManager:
    def test_initialization(self, tmp_path):
        # Test successful initialization
        json_file = create_test_json_file(tmp_path)
        manager = ChunkManager(str(json_file))
        
        assert manager.total_chunks == 2
        assert manager.remaining_chunks == 2
        assert len(manager._processed_set) == 0

    def test_invalid_json_file(self, tmp_path):
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            ChunkManager(str(tmp_path / "nonexistent.json"))
        
        # Test with non-JSON file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a json")
        with pytest.raises(ValueError):
            ChunkManager(str(txt_file))
    
    def test_invalid_version(self, tmp_path):
        # Create file with invalid version
        invalid_data = {
            "version": "invalid_version",
            "chunks": [],
            "summary": {"total_chunks": 0, "processed": 0, "processed_ids": []}
        }
        file_path = tmp_path / "invalid_version.json"
        with open(file_path, 'w') as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(ValueError):
            ChunkManager(str(file_path))
    
    def test_get_next_chunk(self, tmp_path):
        json_file = create_test_json_file(tmp_path)
        manager = ChunkManager(str(json_file))
        
        # Get first chunk
        chunk = manager.get_next_chunk()
        assert isinstance(chunk, pd.DataFrame)
        assert len(chunk) == 2
        assert list(chunk['col1']) == [1, 2]
        
        # Mark first chunk as processed
        manager.mark_chunk_processed()
        
        # Get second chunk
        chunk = manager.get_next_chunk()
        assert len(chunk) == 2
        assert list(chunk['col1']) == [3, 4]
        
        # No more chunks
        manager.mark_chunk_processed()
        assert manager.get_next_chunk() is None
    
    def test_mark_chunk_processed(self, tmp_path):
        json_file = create_test_json_file(tmp_path, processed_ids=[1])
        manager = ChunkManager(str(json_file))
        
        # First chunk is already processed
        assert manager.remaining_chunks == 1
        
        # Process second chunk
        chunk = manager.get_next_chunk()  # This should be chunk 2
        assert chunk is not None  # Make sure we got a chunk
        manager.mark_chunk_processed()
        
        assert manager.remaining_chunks == 0
        
        # Create a new manager to test the error case
        fresh_manager = ChunkManager(str(json_file))
        # Try to mark as processed without getting a chunk first
        with pytest.raises(RuntimeError, match="No chunk to mark as processed"):
            fresh_manager.mark_chunk_processed()
    
    def test_save_state(self, tmp_path):
        json_file = create_test_json_file(tmp_path)
        manager = ChunkManager(str(json_file))
        
        # Process first chunk
        manager.get_next_chunk()
        manager.mark_chunk_processed()
        manager.save_state()
        
        # Reload and verify
        manager2 = ChunkManager(str(json_file))
        assert manager2.remaining_chunks == 1
        assert "1" in manager2._processed_set
    
    def test_repr(self, tmp_path):
        json_file = create_test_json_file(tmp_path, processed_ids=[1])
        manager = ChunkManager(str(json_file))
        
        assert "ChunkManager(" in repr(manager)
        assert "processed=1/2" in repr(manager)
        assert json_file.name in repr(manager)
