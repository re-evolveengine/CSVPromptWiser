import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

class StreamlitDatasetHandler:
    """Handles dataset upload, parsing, and optional saving for Streamlit."""

    def __init__(self, save_dir: str = "data"):
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
