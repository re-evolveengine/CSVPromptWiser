import pytest
import pandas as pd
from pathlib import Path

from cli.dataset_loader import DatasetLoader

# Test data
TEST_DATA = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})


@pytest.fixture
def temp_files(tmp_path):
    """Create temporary CSV and Parquet files for testing."""
    csv_path = tmp_path / "test.csv"
    parquet_path = tmp_path / "test.parquet"

    TEST_DATA.to_csv(csv_path, index=False)
    TEST_DATA.to_parquet(parquet_path)

    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "invalid": str(tmp_path / "test.txt"),
        "missing": str(tmp_path / "missing.csv")
    }


def test_load_csv(temp_files):
    """Test loading a CSV file."""
    loader = DatasetLoader(directory_path=str(Path(temp_files["csv"]).parent))
    df = loader.load_from_upload("test.csv")
    pd.testing.assert_frame_equal(df, TEST_DATA)


def test_load_parquet(temp_files):
    """Test loading a Parquet file."""
    loader = DatasetLoader(directory_path=str(Path(temp_files["parquet"]).parent))
    df = loader.load_from_upload("test.parquet")
    pd.testing.assert_frame_equal(df, TEST_DATA)


def test_invalid_file_type(temp_files, tmp_path):
    """Test handling of unsupported file types."""
    # Create an empty file with unsupported extension
    invalid_file = tmp_path / "test.txt"
    invalid_file.touch()
    
    loader = DatasetLoader(directory_path=str(tmp_path))
    with pytest.raises(ValueError, match="Only .csv and .parquet files are supported"):
        loader.load_from_upload("test.txt")


def test_missing_file(temp_files):
    """Test handling of missing files."""
    loader = DatasetLoader(directory_path=str(Path(temp_files["missing"]).parent))
    with pytest.raises(FileNotFoundError):
        loader.load_from_upload("missing.csv")


def test_missing_filename():
    """Test handling of missing filename."""
    loader = DatasetLoader()
    with pytest.raises(ValueError, match="Please provide a file name"):
        loader.load_from_upload(None)


def test_directory_path_output(capsys):
    """Test directory path output during initialization."""
    loader = DatasetLoader()
    captured = capsys.readouterr()
    output = captured.out.strip()
    assert "Please ensure the dataset file is in the following directory:" in output
    assert "PromptPilot" in output  # Updated to check for PromptPilot in the path