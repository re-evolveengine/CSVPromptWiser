import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from utils.constants import DATA_DIR


class DatasetHandler:
    """Handles dataset upload, parsing, and optional saving for Streamlit."""

    def __init__(self, save_dir: str = DATA_DIR):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.uploaded_file = None
        self.dataframe = None
        self.file_path = None

    def load_from_upload(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load DataFrame from an uploaded CSV or Parquet file."""
        self.uploaded_file = uploaded_file

        if not uploaded_file:
            return None

        try:
            if uploaded_file.name.endswith(".csv"):
                self.dataframe = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                self.dataframe = pd.read_parquet(uploaded_file)
            else:
                st.error("❌ Unsupported file type. Please upload CSV or Parquet.")
                return None
        except Exception as e:
            st.error(f"❌ Failed to load dataset: {e}")
            return None

        return self.dataframe

    def save_uploaded_file(self) -> str:
        """Save the uploaded file to disk and return the file path."""
        if not self.uploaded_file:
            raise ValueError("No file uploaded to save.")

        save_path = self.save_dir / self.uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(self.uploaded_file.getbuffer())

        self.file_path = str(save_path)
        return self.file_path

    def get_saved_file_name(self) -> Optional[str]:
        """Return the name of a saved file if one exists in the save directory."""
        # First check if we have a file path from the current session
        if self.file_path and os.path.exists(self.file_path):
            return Path(self.file_path).name
            
        # If not, check the save directory for any existing files
        try:
            # Get all CSV and Parquet files in the save directory
            saved_files = list(self.save_dir.glob('*.csv')) + list(self.save_dir.glob('*.parquet'))
            if saved_files:
                # Return the most recently modified file
                latest_file = max(saved_files, key=os.path.getmtime)
                self.file_path = str(latest_file)  # Update file_path for future reference
                return latest_file.name
        except Exception as e:
            st.warning(f"Warning checking saved files: {e}")
            
        return None

    def load_saved_file(self, file_name: str) -> Optional[pd.DataFrame]:
        """Load a previously saved CSV or Parquet file from disk."""
        file_path = self.save_dir / file_name
        if not file_path.exists():
            st.error(f"❌ File not found: `{file_path}`")
            return None

        try:
            if file_path.suffix == ".csv":
                self.dataframe = pd.read_csv(file_path)
            elif file_path.suffix == ".parquet":
                self.dataframe = pd.read_parquet(file_path)
            else:
                st.error("❌ Unsupported file type.")
                return None

            self.file_path = str(file_path)
            return self.dataframe
        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
            return None
