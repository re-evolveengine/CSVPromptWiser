import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import io

from streamlit_dir.stramlit_dataset_handler import StreamlitDatasetHandler

# Sample data for testing
SAMPLE_CSV = """name,age,city
Alice,28,New York
Bob,34,Los Angeles
Charlie,45,Chicago"""

SAMPLE_DATA = [
    {"name": "Alice", "age": 28, "city": "New York"},
    {"name": "Bob", "age": 34, "city": "Los Angeles"},
    {"name": "Charlie", "age": 45, "city": "Chicago"}
]

@pytest.fixture
def handler(tmp_path):
    """Fixture that provides a StreamlitDatasetHandler with a temporary directory."""
    return StreamlitDatasetHandler(save_dir=str(tmp_path))

class TestStreamlitDatasetHandler:
    def test_initialization(self, tmp_path):
        """Test that the handler initializes with the correct save directory."""
        handler = StreamlitDatasetHandler(save_dir=str(tmp_path))
        assert handler.save_dir == Path(tmp_path)
        assert handler.uploaded_file is None
        assert handler.dataframe is None
        assert handler.file_path is None

    def test_load_csv(self, handler):
        """Test loading a CSV file."""
        # Create a mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_file.read.return_value = SAMPLE_CSV.encode()

        # Mock pd.read_csv to return our sample data
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(SAMPLE_DATA)
            result = handler.load_from_upload(mock_file)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["name", "age", "city"]
        assert handler.uploaded_file == mock_file
        assert handler.dataframe is not None

    def test_load_parquet(self, handler):
        """Test loading a Parquet file."""
        # Create a mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test.parquet"

        # Mock pd.read_parquet to return our sample data
        with patch('pandas.read_parquet') as mock_read_parquet:
            mock_read_parquet.return_value = pd.DataFrame(SAMPLE_DATA)
            result = handler.load_from_upload(mock_file)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["name", "age", "city"]

    def test_load_unsupported_file_type(self, handler):
        """Test loading an unsupported file type."""
        # Create a mock uploaded file with unsupported extension
        mock_file = MagicMock()
        mock_file.name = "test.txt"

        # Mock st.error to verify it's called
        with patch('streamlit.error') as mock_error:
            result = handler.load_from_upload(mock_file)

            assert result is None
            mock_error.assert_called_once_with("❌ Unsupported file type. Please upload CSV or Parquet.")

    def test_save_uploaded_file(self, handler, tmp_path):
        """Test saving an uploaded file."""
        # Create a mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_file.getbuffer.return_value = io.BytesIO(SAMPLE_CSV.encode()).getbuffer()

        # Set the uploaded file
        handler.uploaded_file = mock_file

        # Save the file
        save_path = handler.save_uploaded_file()

        # Verify the file was saved
        expected_path = tmp_path / "test.csv"
        assert save_path == str(expected_path)
        assert os.path.exists(expected_path)
        assert handler.file_path == str(expected_path)

        # Verify file contents
        with open(expected_path, 'r') as f:
            content = f.read()
        assert content == SAMPLE_CSV

    def test_save_without_uploaded_file(self, handler):
        """Test that saving without an uploaded file raises an error."""
        with pytest.raises(ValueError, match="No file uploaded to save."):
            handler.save_uploaded_file()

    def test_load_none_file(self, handler):
        """Test loading when no file is uploaded."""
        result = handler.load_from_upload(None)
        assert result is None
