# core/chunker.py
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from math import ceil


class DataFrameChunker:
    """Handles DataFrame chunking and serialization separately."""

    def __init__(self, chunk_size: int = 1000):
        """
        Args:
            chunk_size: Default rows per chunk (configurable per operation)
        """
        self.chunk_size = chunk_size
        self._chunks: List[pd.DataFrame] = []

    def chunk_dataframe(self,
                        df: pd.DataFrame,
                        chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """
        Split DataFrame into chunks.

        Args:
            df: DataFrame to split
            chunk_size: Optional override of default chunk size

        Returns:
            List of DataFrame chunks (also stored in .chunks property)
        """
        size = chunk_size or self.chunk_size
        if size <= 0:
            raise ValueError("Chunk size must be positive")

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
    def save_chunks_to_json(chunks: List[pd.DataFrame],
                            file_path: str,
                            max_rows_per_chunk: Optional[int] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save chunks to JSON file with configurable options.

        Args:
            chunks: List of DataFrames to save
            file_path: Output JSON path
            max_rows_per_chunk: Limit rows saved per chunk (None for all)
            metadata: Optional metadata to include

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("No chunks to save")

        output = {
            "metadata": metadata or {},
            "chunks": [],
            "summary": {
                "total_chunks": len(chunks),
                "saved_rows": 0
            }
        }

        for i, df in enumerate(chunks):
            chunk_data = df.head(max_rows_per_chunk) if max_rows_per_chunk else df
            output["chunks"].append({
                "chunk_number": i + 1,
                "data": chunk_data.to_dict(orient="records"),
                "original_rows": len(df),
                "saved_rows": len(chunk_data)
            })
            output["summary"]["saved_rows"] += len(chunk_data)

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved {len(chunks)} chunks to {file_path} "
              f"({output['summary']['saved_rows']} total rows)")