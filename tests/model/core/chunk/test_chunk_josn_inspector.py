import json
import tempfile
import pytest
import types
from pathlib import Path
import streamlit as st

# Mock Streamlit secrets
st.secrets = types.SimpleNamespace()
st.secrets.is_local = True

from model.core.chunk.chunk_json_inspector import ChunkJSONInspector


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def make_chunk_file(path: Path, name: str, valid=True, processed_ids=None):
    """Helper to create a JSON chunk file."""
    processed_ids = processed_ids or []
    if valid:
        content = {
            "version": "1.0",
            "summary": {
                "total_chunks": 3,
                "chunk_size": 2,
                "processed_ids": processed_ids
            },
            "chunks": [
                {"chunk_id": "c1", "data": [1, 2]},
                {"chunk_id": "c2", "data": [3, 4]},
                {"chunk_id": "c3", "data": [5, 6]},
            ]
        }
    else:
        content = {"foo": "bar"}  # Clearly invalid
    file_path = path / name
    file_path.write_text(json.dumps(content))
    return file_path


def test_init_creates_directory_when_parent_exists(temp_dir):
    new_dir = temp_dir / "child"
    inspector = ChunkJSONInspector(str(new_dir))
    assert new_dir.exists()
    assert inspector.directory == new_dir.resolve()


def test_init_raises_if_parent_missing(tmp_path):
    non_existent = tmp_path / "nonexistent" / "child"
    with pytest.raises(FileNotFoundError):
        ChunkJSONInspector(str(non_existent))


def test_find_valid_chunk_file_finds_first_valid(temp_dir):
    # Create invalid file first
    make_chunk_file(temp_dir, "invalid.json", valid=False)
    valid_file = make_chunk_file(temp_dir, "valid.json", valid=True)
    inspector = ChunkJSONInspector(str(temp_dir))
    found = inspector.find_valid_chunk_file()
    assert found and found.samefile(valid_file)



def test_find_valid_chunk_file_returns_none_if_no_valid(temp_dir):
    make_chunk_file(temp_dir, "bad.json", valid=False)
    inspector = ChunkJSONInspector(str(temp_dir))
    assert inspector.find_valid_chunk_file() is None


def test_inspect_chunk_file_counts_processed_correctly(temp_dir):
    file_path = make_chunk_file(temp_dir, "chunks.json", valid=True, processed_ids=["c1"])
    inspector = ChunkJSONInspector(str(temp_dir))
    summary = inspector.inspect_chunk_file(file_path)
    assert summary["total_chunks"] == 3
    assert summary["processed_chunks"] == 1
    assert summary["unprocessed_chunks"] == 2
    assert summary["can_resume"] is True
    assert summary["chunk_size"] == 2
    assert summary["file"] == "chunks.json"
    assert summary["version"] == "1.0"


def test_is_valid_chunk_json_static_method():
    good_data = {
        "summary": {"total_chunks": 1},
        "chunks": [{"chunk_id": "a", "data": []}]
    }
    bad_data = {"summary": {}, "chunks": []}
    assert ChunkJSONInspector._is_valid_chunk_json(good_data) is True
    assert ChunkJSONInspector._is_valid_chunk_json(bad_data) is False
