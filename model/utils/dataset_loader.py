from pathlib import Path
import os
from typing import Optional

import pandas as pd

# Get the path to the user's Documents directory
DOCUMENTS_PATH = os.path.join(os.path.expanduser("~"), "Documents")


class DatasetLoader:
    """Minimal dataset loader for CSV and Parquet files."""

    def __init__(self, directory_path: str = DOCUMENTS_PATH):
        self.file_path = None
        self.directory_path = directory_path
        print("Please ensure the dataset file is in the following directory:")
        print(DOCUMENTS_PATH)

    def load(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """Load CSV/Parquet file. Falls back to default path if none provided."""

        if not file_name:
            raise ValueError("Please provide a file name")
        else:
            self.file_path = os.path.join(self.directory_path, file_name)

        path = Path(self.file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(path)
        else:
            raise ValueError("Only .csv and .parquet files are supported")