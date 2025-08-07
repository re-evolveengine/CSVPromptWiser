import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4
import pandas as pd

from model.utils.constants import JSON_CHUNK_VERSION, DEFAULT_CHUNK_SIZE, TEMP_DIR


class DataFrameChunker:
    """Handles DataFrame chunking and JSON serialization only."""

    def __init__(self, chunk_size: Optional[int] = None):
        """
        Args:
            chunk_size: Number of rows per chunk. If None, <= 0, or not provided,
                      uses DEFAULT_CHUNK_SIZE.
        """
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else DEFAULT_CHUNK_SIZE
        self._chunks: List[pd.DataFrame] = []

    def chunk_dataframe(
            self,
            df: pd.DataFrame,
            chunk_size: Optional[int] = None
    ) -> List[pd.DataFrame]:
        """
        Split DataFrame into smaller chunks and add a unique source_id per row.

        Args:
            df: Input DataFrame to split
            chunk_size: Optional override for chunk size. If None or <= 0, uses the instance's chunk_size.

        Returns:
            List of DataFrame chunks
        """
        # Use the provided chunk_size if valid, otherwise use the instance's chunk_size
        size = chunk_size if chunk_size and chunk_size > 0 else self.chunk_size

        # Handle empty DataFrame
        if df.empty:
            self._chunks = []
            return self._chunks

        # ✅ NEW: Add a UUID to each row as 'source_id'
        df = df.copy()
        df["source_id"] = [str(uuid4()) for _ in range(len(df))]

        total_rows = len(df)
        self._chunks = [
            df.iloc[i:i + size]
            for i in range(0, total_rows, size)
        ]

        print(f"Split {total_rows:,} rows into {len(self._chunks)} "
              f"chunks of ~{size:,} rows each")
        return self._chunks

    @property
    def chunks(self) -> List[pd.DataFrame]:
        """Access stored chunks."""
        if not self._chunks:
            raise ValueError("No chunks available - run chunk_dataframe() first")
        return self._chunks

    def save_chunks_to_json(
            self,
            chunks: List[pd.DataFrame],
            file_path: str = os.path.join(TEMP_DIR, "chunks.json"),
            max_rows_per_chunk: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Safely save chunks to JSON with versioning and atomic writes.

        Args:
            chunks: List of DataFrames to save
            file_path: Output JSON path
            max_rows_per_chunk: Max rows per chunk (None = all)
            metadata: Optional metadata to include

        Raises:
            ValueError: If chunks list is empty
            OSError: If file operations fail
        """
        if not chunks:
            raise ValueError("No chunks to save")

        output = {
            "version": JSON_CHUNK_VERSION,
            "metadata": metadata or {},
            "chunks": [],
            "summary": {
                "total_chunks": len(chunks),
                "processed_ids": [],  # Track processed UUIDs
                "chunk_size": self.chunk_size
            }
        }

        for df in chunks:
            chunk_data = df.head(max_rows_per_chunk) if max_rows_per_chunk else df
            output["chunks"].append({
                "chunk_id": str(uuid4()),
                "data": chunk_data.to_dict(orient='records'),
                "original_rows": len(df)
            })

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix('.tmp')

        try:
            with open(temp_path, 'w') as f:
                json.dump(output, f, indent=2)
            os.replace(temp_path, path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise OSError(f"Failed to save chunks: {str(e)}") from e

        print(f"Saved {len(chunks)} chunks to {file_path}")
