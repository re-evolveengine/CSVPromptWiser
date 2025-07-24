# core/chunker.py
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4
import pandas as pd


class DataFrameChunker:
    """Handles DataFrame chunking and JSON serialization only."""

    # Current version of the JSON format
    JSON_VERSION = 1.0

    def __init__(self, chunk_size: int = 1000):
        """
        Args:
            chunk_size: Default number of rows per chunk
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        self.chunk_size = chunk_size
        self._chunks: List[pd.DataFrame] = []

    def chunk_dataframe(self,
                        df: pd.DataFrame,
                        chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """
        Split DataFrame into smaller chunks.

        Args:
            df: Input DataFrame to split
            chunk_size: Optional override for chunk size. If <= 0, uses default chunk size.

        Returns:
            List of DataFrame chunks
        """
        if chunk_size is None:
            size = self.chunk_size
        else:
            # Use default chunk size if provided size is invalid
            size = self.chunk_size if chunk_size <= 0 else chunk_size

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

    @staticmethod
    def save_chunks_to_json(
            chunks: List[pd.DataFrame],
            file_path: str,
            max_rows_per_chunk: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Safely save chunks to JSON with versioning and atomic writes.

        Args:
            chunks: List of DataFrames to save
            file_path: Output JSON path
            max_rows_per_chunk: Max rows per chunk (None for all)
            metadata: Optional metadata to include

        Raises:
            ValueError: If chunks list is empty
            OSError: If file operations fail
        """
        if not chunks:
            raise ValueError("No chunks to save")

        # Prepare output structure with versioning
        output = {
            "version": 1.0,
            "metadata": metadata or {},
            "chunks": [],
            "summary": {
                "total_chunks": len(chunks),
                "saved_rows": 0,
                "processed": 0  # Track processed chunks
            }
        }

        # Add chunks with unique IDs
        for idx, df in enumerate(chunks):
            chunk_data = df.head(max_rows_per_chunk) if max_rows_per_chunk else df
            output["chunks"].append({
                "chunk_number": idx + 1,
                "data": chunk_data.to_dict(orient='records'),
                "original_rows": len(df),
                "saved_rows": len(chunk_data)
            })
            output["summary"]["saved_rows"] += len(chunk_data)

        # Atomic write operation
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(output, f, indent=2)

            # Atomic replace
            os.replace(temp_path, path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise OSError(f"Failed to save chunks: {str(e)}") from e

        print(f"Saved {len(chunks)} chunks to {file_path} "
              f"({output['summary']['saved_rows']} total rows)")