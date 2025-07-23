# core/chunk_manager.py
import json
from typing import Any, Callable, Optional, Dict, List
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class ChunkManager:
    """Manages sequential processing of chunks from JSON storage."""

    def __init__(self, json_path: str):
        """
        Args:
            json_path: Path to JSON file created by DataFrameChunker.save_chunks_to_json()
        """
        self.json_path = Path(json_path)
        self._validate_json_file()

        with open(self.json_path) as f:
            self.metadata = json.load(f)

        self._current_chunk = 0
        self._processed_chunks = 0

    def _validate_json_file(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        if self.json_path.suffix != '.json':
            raise ValueError("File must be JSON format")

    @property
    def total_chunks(self) -> int:
        """Total chunks available in the file."""
        return self.metadata['summary']['total_chunks']

    @property
    def remaining_chunks(self) -> int:
        """Chunks left to process."""
        return self.total_chunks - self._processed_chunks

    def get_next_chunk(self) -> Optional[pd.DataFrame]:
        """Loads and returns the next chunk as DataFrame."""
        if self._current_chunk >= self.total_chunks:
            return None

        chunk_data = self.metadata['chunks'][self._current_chunk]
        self._current_chunk += 1
        self._processed_chunks += 1

        return pd.DataFrame(chunk_data['data'])

    def process_chunks(self,
                       func: Callable[[pd.DataFrame], Any],
                       show_progress: bool = True) -> List[Any]:
        """
        Processes all chunks sequentially.

        Args:
            func: Function that takes a DataFrame and returns processed result
            show_progress: Whether to show progress bar

        Returns:
            List of results in chunk order
        """
        results = []
        iterator = range(self.remaining_chunks)

        if show_progress:
            iterator = tqdm(iterator,
                            desc=f"Processing {self.json_path.name}",
                            unit="chunk")

        for _ in iterator:
            chunk = self.get_next_chunk()
            results.append(func(chunk))

        return results

    def __repr__(self) -> str:
        return (f"ChunkManager(file='{self.json_path.name}', "
                f"processed={self._processed_chunks}/"
                f"{self.total_chunks})")