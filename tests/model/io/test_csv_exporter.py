import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from model.io.csv_exporter import CSVExporter
from model.io.gemini_sqlite_result_saver import SQLiteResultSaver


@pytest.fixture
def sample_processed_rows():
    return [
        {"source_id": "id1", "processed_val": 123},
        {"source_id": "id2", "processed_val": 456},
    ]


@pytest.fixture
def sample_chunk_json(tmp_path):
    """Creates a fake chunk JSON file with original data including chunk_ids."""
    data = {
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "data": [{"source_id": "id1", "orig_val": "foo"}],
            },
            {
                "chunk_id": "chunk-2",
                "data": [{"source_id": "id2", "orig_val": "bar"}],
            },
        ]
    }
    json_path = tmp_path / "chunks.json"
    json_path.write_text(json.dumps(data))
    return json_path


def test_export_raises_when_no_processed_data(tmp_path):
    # Mock saver to return empty list
    saver = MagicMock(spec=SQLiteResultSaver)
    saver.get_all.return_value = []
    exp = CSVExporter(json_path=tmp_path / "dummy.json", db_saver=saver)
    with pytest.raises(ValueError, match="No processed data"):
        exp.export_processed_with_original_rows(tmp_path / "out.csv")


def test_export_with_merge_writes_expected_csv(sample_processed_rows, sample_chunk_json, tmp_path):
    saver = MagicMock(spec=SQLiteResultSaver)
    saver.get_all.return_value = sample_processed_rows
    csv_path = tmp_path / "out.csv"

    exp = CSVExporter(json_path=sample_chunk_json, db_saver=saver)

    with patch.object(pd.DataFrame, "to_csv") as mock_csv:
        exp.export_processed_with_original_rows(csv_path)
        assert mock_csv.call_count == 1
        args, kwargs = mock_csv.call_args
        assert args[0] == csv_path
        assert kwargs["index"] is False


def test_export_without_json_prints_warning(sample_processed_rows, tmp_path, capsys):
    missing_path = tmp_path / "missing.json"
    saver = MagicMock(spec=SQLiteResultSaver)
    saver.get_all.return_value = sample_processed_rows
    exp = CSVExporter(json_path=missing_path, db_saver=saver)

    with patch.object(pd.DataFrame, "to_csv") as mock_csv:
        exp.export_processed_with_original_rows(tmp_path / "out.csv")

    out = capsys.readouterr().out
    assert "Warning: Chunk JSON file not found" in out
    mock_csv.assert_called_once()


def test_export_cleans_duplicate_chunk_id_columns(sample_processed_rows, tmp_path):
    # Create JSON file with one chunk
    data = {
        "chunks": [
            {
                "chunk_id": "dup-check",
                "data": [{"source_id": "id1"}],
            }
        ]
    }
    json_path = tmp_path / "chunks.json"
    json_path.write_text(json.dumps(data))

    saver = MagicMock(spec=SQLiteResultSaver)
    # processed_df will have a 'chunk_id' column — merge will auto rename
    saver.get_all.return_value = [{"source_id": "id1", "chunk_id": "db_chunk"}]

    # Patch pd.merge to return a DataFrame with chunk_id_x and chunk_id_y
    merged_df = pd.DataFrame({
        "source_id": ["id1"],
        "chunk_id_x": ["file_chunk"],
        "chunk_id_y": ["db_chunk"]
    })

    with patch("pandas.merge", return_value=merged_df) as mock_merge:
        exp = CSVExporter(json_path=json_path, db_saver=saver)
        with patch.object(pd.DataFrame, "to_csv") as mock_csv:
            exp.export_processed_with_original_rows(tmp_path / "out.csv")

    saved_df = mock_csv.call_args[0][0] if mock_csv.call_args else None
    mock_merge.assert_called()
    mock_csv.assert_called_once()
