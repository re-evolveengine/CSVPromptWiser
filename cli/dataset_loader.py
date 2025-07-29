from pathlib import Path
import os
from typing import Optional

import pandas as pd

from model.utils.constants import DATA_DIR_CLI


class DatasetLoader:
    """Minimal dataset loader for CSV and Parquet files."""

    def __init__(self, directory_path: str = DATA_DIR_CLI):
        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        self.file_path = None
        self.directory_path = directory_path
        print("Please ensure the dataset file is in the following directory:")
        print(self.directory_path)

    def load_from_upload(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load a dataset from a file.
        
        Args:
            file_name: Name of the file to load (must be .csv or .parquet)
            
        Returns:
            pd.DataFrame: The loaded data
            
        Raises:
            ValueError: If no file name is provided or file type is not supported
            FileNotFoundError: If the specified file doesn't exist
        """
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