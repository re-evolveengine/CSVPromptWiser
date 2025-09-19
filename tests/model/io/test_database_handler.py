import io
import os
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from model.io.dataset_handler import DatasetHandler


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    return tmp_path


@pytest.fixture
def handler(tmp_dataset_dir):
    return DatasetHandler(save_dir=tmp_dataset_dir)


def make_uploaded_file(name: str, content: bytes = b"col1,col2\n1,2\n") -> MagicMock:
    """Creates a fake Streamlit UploadedFile-like object."""
    mock_file = MagicMock()
    mock_file.name = name
    mock_file.getbuffer.return_value = content
    # For pandas.read_csv to work, .read and seek are needed
    mock_file.read = io.BytesIO(content).read
    mock_file.seek = io.BytesIO(content).seek
    return mock_file


def test_load_from_upload_csv(handler):
    uploaded = make_uploaded_file("test.csv")
    with patch("pandas.read_csv", return_value=pd.DataFrame({"a": [1]})) as mock_read:
        df = handler.load_from_upload(uploaded)
        assert isinstance(df, pd.DataFrame)
        mock_read.assert_called_once()


def test_load_from_upload_parquet(handler):
    uploaded = make_uploaded_file("test.parquet")
    with patch("pandas.read_parquet", return_value=pd.DataFrame({"a": [1]})) as mock_read:
        df = handler.load_from_upload(uploaded)
        assert isinstance(df, pd.DataFrame)
        mock_read.assert_called_once()


def test_load_from_upload_unsupported_and_error(handler):
    uploaded = make_uploaded_file("file.txt")
    with patch("streamlit.error") as mock_err:
        result = handler.load_from_upload(uploaded)
        assert result is None
        mock_err.assert_called_once()

    with patch("pandas.read_csv", side_effect=Exception("fail")), \
         patch("streamlit.error") as mock_err:
        bad_csv = make_uploaded_file("file.csv")
        assert handler.load_from_upload(bad_csv) is None
        assert mock_err.call_count == 1


def test_save_uploaded_file_success(tmp_path, handler):
    uploaded = make_uploaded_file("save.csv", b"abc")
    handler.uploaded_file = uploaded
    file_path = handler.save_uploaded_file()
    assert Path(file_path).exists()
    assert Path(file_path).read_bytes() == b"abc"


def test_save_uploaded_file_no_upload(handler):
    handler.uploaded_file = None
    with pytest.raises(ValueError):
        handler.save_uploaded_file()


def test_get_saved_file_name_from_current_session(tmp_path, handler):
    fake_file = tmp_path / "file.csv"
    fake_file.write_text("hi")
    handler.file_path = str(fake_file)
    assert handler.get_saved_file_name() == "file.csv"


def test_get_saved_file_name_from_directory(tmp_path, handler):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.parquet"
    f1.write_text("1")
    f2.write_text("2")
    # Make f2 newer
    os.utime(f2, None)
    handler.file_path = None
    name = handler.get_saved_file_name()
    assert name in ("a.csv", "b.parquet")


def test_get_saved_file_name_handles_exception(tmp_path, handler):
    # Break .glob
    handler.save_dir = "nonexistent_dir"
    with patch("streamlit.warning") as mock_warn:
        assert handler.get_saved_file_name() is None
        mock_warn.assert_called_once()


def test_load_saved_file_csv_and_parquet(tmp_path, handler):
    csv_file = tmp_path / "f.csv"
    csv_file.write_text("a,b\n1,2")
    parq_file = tmp_path / "f.parquet"
    parq_file.write_text("fake")

    with patch("pandas.read_csv", return_value=pd.DataFrame({"x": [1]})) as mock_csv:
        df = handler.load_saved_file(str(csv_file.name))
        assert isinstance(df, pd.DataFrame)
        mock_csv.assert_called_once()

    with patch("pandas.read_parquet", return_value=pd.DataFrame({"x": [1]})) as mock_parq:
        df = handler.load_saved_file(str(parq_file.name))
        assert isinstance(df, pd.DataFrame)
        mock_parq.assert_called_once()


def test_load_saved_file_not_found_or_unsupported(handler):
    with patch("streamlit.error") as mock_err:
        assert handler.load_saved_file("no_file.csv") is None
        assert mock_err.call_count == 1

    tmp_file = handler.save_dir / "bad.txt"
    tmp_file.write_text("data")
    with patch("streamlit.error") as mock_err:
        assert handler.load_saved_file(tmp_file.name) is None
        assert mock_err.call_count == 1


def test_load_saved_file_read_error(tmp_path, handler):
    csv_path = tmp_path / "f.csv"
    csv_path.write_text("a,b\n1,2")
    handler.save_dir = tmp_path
    with patch("pandas.read_csv", side_effect=Exception("boom")), \
         patch("streamlit.error") as mock_err:
        assert handler.load_saved_file(csv_path.name) is None
        mock_err.assert_called_once()
