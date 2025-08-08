import json
import os
import tempfile
from pathlib import Path
import pytest

from model.core.chunk.chunk_json_inspector import ChunkJSONInspector


@pytest.fixture
def temp_dir():
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    yield test_dir
    # Cleanup after test
    for file in Path(test_dir).glob('*'):
        try:
            file.unlink()
        except Exception:
            pass
    try:
        os.rmdir(test_dir)
    except Exception:
        pass


@pytest.fixture
def valid_chunk_data():
    return {
        "version": "1.0",
        "chunks": [
            {"chunk_id": 1, "data": "chunk1"},
            {"chunk_id": 2, "data": "chunk2"},
            {"chunk_id": 3, "data": "chunk3"}
        ],
        "summary": {
            "total_chunks": 3,
            "processed_ids": [1, 2]
        }
    }


@pytest.fixture
def test_file(temp_dir, valid_chunk_data):
    # Create a test JSON file
    test_file = os.path.join(temp_dir, "test_chunks.json")
    with open(test_file, 'w') as f:
        json.dump(valid_chunk_data, f)
    return test_file


def test_init_creates_directory_if_not_exists(temp_dir):
    # Create a new directory path that doesn't exist
    new_dir = os.path.join(temp_dir, "new_dir")
    inspector = ChunkJSONInspector(directory_path=new_dir)
    assert os.path.exists(new_dir)


def test_init_raises_for_nonexistent_directory():
    """Test that initializing with a non-existent parent directory raises an error."""
    # Create a unique path that definitely doesn't exist
    import uuid
    non_existent_path = str(Path("/tmp") / f"nonexistent_{uuid.uuid4()}" / "test_dir")
    
    # Ensure the parent doesn't exist
    parent_path = Path(non_existent_path).parent
    if parent_path.exists():
        parent_path.rmdir()
    
    # Verify the parent doesn't exist
    assert not parent_path.exists(), f"Parent path {parent_path} exists when it shouldn't"
    
    # This should raise FileNotFoundError because the parent doesn't exist
    with pytest.raises(FileNotFoundError, match=f"Parent directory does not exist"):
        ChunkJSONInspector(directory_path=non_existent_path)


def test_find_valid_chunk_file_finds_valid_file(temp_dir, test_file):
    inspector = ChunkJSONInspector(directory_path=temp_dir)
    result = inspector.find_valid_chunk_file()
    assert result is not None
    assert os.path.basename(result) == "test_chunks.json"


def test_find_valid_chunk_file_returns_none_for_invalid_files(temp_dir, test_file):
    # Create an invalid JSON file
    invalid_file = os.path.join(temp_dir, "invalid.json")
    with open(invalid_file, 'w') as f:
        f.write("not a valid json")

    inspector = ChunkJSONInspector(directory_path=temp_dir)
    result = inspector.find_valid_chunk_file()
    # Should still find the valid file, not the invalid one
    assert result is not None
    assert os.path.basename(result) != "invalid.json"


def test_inspect_chunk_file_returns_correct_summary(test_file, valid_chunk_data):
    inspector = ChunkJSONInspector()
    result = inspector.inspect_chunk_file(Path(test_file))

    assert result["file"] == "test_chunks.json"
    assert result["version"] == "1.0"
    assert result["total_chunks"] == 3
    assert result["processed_chunks"] == 2
    assert result["unprocessed_chunks"] == 1  # Only chunk 3 is unprocessed
    assert result["can_resume"] is True


def test_is_valid_chunk_json_validation(valid_chunk_data):
    # Test valid data
    assert ChunkJSONInspector._is_valid_chunk_json(valid_chunk_data) is True

    # Test missing chunks
    invalid_data = valid_chunk_data.copy()
    del invalid_data["chunks"]
    assert ChunkJSONInspector._is_valid_chunk_json(invalid_data) is False

    # Test missing summary
    invalid_data = valid_chunk_data.copy()
    del invalid_data["summary"]
    assert ChunkJSONInspector._is_valid_chunk_json(invalid_data) is False

    # Test invalid chunks structure
    invalid_data = valid_chunk_data.copy()
    invalid_data["chunks"] = [{"no_chunk_id": 1}]
    assert ChunkJSONInspector._is_valid_chunk_json(invalid_data) is False
