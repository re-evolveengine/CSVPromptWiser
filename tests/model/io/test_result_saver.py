import json
import os
import pandas as pd
import pytest
from pathlib import Path
from unittest import mock

from model.io.result_saver import ResultSaver


@pytest.fixture
def temp_dir(tmp_path):
    """Temp dir for saving files."""
    return tmp_path


def make_df_chunk():
    """A small pandas DataFrame chunk."""
    return pd.DataFrame([{"a": 1}, {"a": 2}])


def make_list_chunk():
    """A chunk as a list of dicts."""
    return [{"a": 1}, {"a": 2}]


# ----------------------
# save_results_to_json
# ----------------------

def test_save_json_with_dataframe_chunk(temp_dir):
    file_path = temp_dir / "out.json"
    results = [{
        "chunk": make_df_chunk(),
        "prompt": "hello model",
        "response": "world"
    }]

    ResultSaver.save_results_to_json(results, str(file_path), metadata={"source": "test"})

    assert file_path.exists()

    data = json.loads(file_path.read_text())
    assert data["version"]  # comes from utils.constants.JSON_CHUNK_VERSION
    assert data["metadata"] == {"source": "test"}
    assert len(data["chunks"]) == 1
    chunk_entry = data["chunks"][0]
    assert chunk_entry["prompt"] == "hello model"
    assert chunk_entry["response"] == "world"
    assert chunk_entry["original_rows"] == 2
    assert isinstance(chunk_entry["chunk_id"], str)
    assert "timestamp" in chunk_entry
    # All processed_ids should match the chunk_id
    assert chunk_entry["chunk_id"] in data["summary"]["processed_ids"]


def test_save_json_with_list_of_dicts_chunk(temp_dir):
    file_path = temp_dir / "out.json"
    results = [{
        "chunk": make_list_chunk(),
        "prompt": "hi",
        "response": "there"
    }]
    ResultSaver.save_results_to_json(results, str(file_path))
    data = json.loads(file_path.read_text())
    assert data["chunks"][0]["data"] == make_list_chunk()
    assert data["chunks"][0]["original_rows"] == 2


def test_save_json_invalid_chunk_type_raises(temp_dir):
    file_path = temp_dir / "out.json"
    results = [{
        "chunk": "not_a_dataframe_or_list",
        "prompt": "hi",
        "response": "there"
    }]
    with pytest.raises(ValueError, match="Chunk must be a pandas DataFrame or a list of dictionaries"):
        ResultSaver.save_results_to_json(results, str(file_path))


def test_save_json_empty_results_raises(temp_dir):
    file_path = temp_dir / "out.json"
    with pytest.raises(ValueError, match="No results to save"):
        ResultSaver.save_results_to_json([], str(file_path))


def test_save_json_oserror_cleanup(temp_dir, monkeypatch):
    file_path = temp_dir / "out.json"
    results = [{
        "chunk": make_df_chunk(),
        "prompt": "p",
        "response": "r"
    }]
    # Force open() to fail
    monkeypatch.setattr("builtins.open", mock.MagicMock(side_effect=OSError("disk error")))
    with pytest.raises(OSError, match="Failed to save results"):
        ResultSaver.save_results_to_json(results, str(file_path))
    # Ensure .tmp file was removed
    tmp_file = file_path.with_suffix(".tmp")
    assert not tmp_file.exists()


# ----------------------
# save_results_to_csv
# ----------------------

def test_save_csv_with_dataframe_chunk(temp_dir):
    file_path = temp_dir / "out.csv"
    results = [{
        "chunk": make_df_chunk(),
        "prompt": "p1",
        "response": "r1"
    }]
    ResultSaver.save_results_to_csv(results, str(file_path))
    assert file_path.exists()
    df = pd.read_csv(file_path)
    assert set(df.columns) == {"a", "chunk_id", "prompt", "response", "timestamp"}
    assert len(df) == 2
    # Ensure prompt/response same for all rows
    assert all(df["prompt"] == "p1")
    assert all(df["response"] == "r1")


def test_save_csv_with_list_of_dicts_chunk(temp_dir):
    file_path = temp_dir / "out.csv"
    results = [{
        "chunk": make_list_chunk(),
        "prompt": "p2",
        "response": "r2"
    }]
    ResultSaver.save_results_to_csv(results, str(file_path))
    df = pd.read_csv(file_path)
    assert list(df["a"]) == [1, 2]
    assert all(df["prompt"] == "p2")


def test_save_csv_invalid_chunk_type(temp_dir):
    file_path = temp_dir / "out.csv"
    results = [{
        "chunk": "not a chunk",
        "prompt": "p",
        "response": "r"
    }]
    with pytest.raises(ValueError, match="Invalid chunk format at index 0"):
        ResultSaver.save_results_to_csv(results, str(file_path))


def test_save_csv_empty_results_raises(temp_dir):
    file_path = temp_dir / "out.csv"
    with pytest.raises(ValueError, match="No results to save"):
        ResultSaver.save_results_to_csv([], str(file_path))
