import os
import json
import tempfile
import types
import pandas as pd
import pytest
import streamlit as st
from uuid import UUID
from pathlib import Path

# Mock Streamlit secrets
st.secrets = types.SimpleNamespace()
st.secrets.is_local = True

from model.core.chunk.chunker import DataFrameChunker


@pytest.fixture
def sample_df():
    """A simple DataFrame for chunking tests."""
    return pd.DataFrame({
        "col1": range(1, 6),
        "col2": list("abcde")
    })


@pytest.fixture
def temp_json_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir) / "chunks.json"


def test_chunk_dataframe_splits_and_adds_source_ids(sample_df):
    chunker = DataFrameChunker(chunk_size=2)
    chunks = chunker.chunk_dataframe(sample_df)

    # Should have split into ceil(5/2) = 3 chunks
    assert len(chunks) == 3
    # Each chunk should be a DataFrame with 'source_id'
    for chunk in chunks:
        assert "source_id" in chunk.columns
        # source_id values should be valid UUIDs
        for sid in chunk["source_id"]:
            UUID(sid)  # Will raise if not a valid UUID

def test_chunk_dataframe_empty_df_returns_empty_list():
    df = pd.DataFrame(columns=["col1", "col2"])
    chunker = DataFrameChunker(chunk_size=2)
    chunks = chunker.chunk_dataframe(df)
    assert chunks == []

def test_chunks_property_returns_stored_chunks(sample_df):
    chunker = DataFrameChunker(chunk_size=3)
    chunks = chunker.chunk_dataframe(sample_df)
    assert chunker.chunks == chunks

def test_chunks_property_raises_if_no_chunks():
    chunker = DataFrameChunker()
    with pytest.raises(ValueError, match="No chunks available"):
        _ = chunker.chunks

def test_save_chunks_to_json_and_load_back(sample_df, temp_json_path):
    chunker = DataFrameChunker(chunk_size=2)
    chunks = chunker.chunk_dataframe(sample_df)

    chunker.save_chunks_to_json(chunks, file_path=str(temp_json_path), metadata={"test": True})
    assert temp_json_path.exists()

    # Load JSON and validate structure
    with open(temp_json_path, "r") as f:
        data = json.load(f)

    assert "version" in data
    assert "chunks" in data
    assert data["summary"]["total_chunks"] == len(chunks)
    assert all("chunk_id" in chunk for chunk in data["chunks"])
    assert all("data" in chunk for chunk in data["chunks"])
    assert data["metadata"]["test"] is True

def test_save_chunks_to_json_raises_on_empty(sample_df, temp_json_path):
    chunker = DataFrameChunker()
    with pytest.raises(ValueError, match="No chunks to save"):
        chunker.save_chunks_to_json([], file_path=str(temp_json_path))

def test_save_chunks_to_json_creates_parent_dirs(sample_df):
    chunker = DataFrameChunker()
    chunks = chunker.chunk_dataframe(sample_df)
    deep_path = Path(tempfile.gettempdir()) / "deep/nested/path/chunks.json"
    try:
        chunker.save_chunks_to_json(chunks, file_path=str(deep_path))
        assert deep_path.exists()
    finally:
        # Cleanup
        if deep_path.exists():
            os.remove(deep_path)
