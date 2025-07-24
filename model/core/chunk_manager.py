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
            self.data = json.load(f)

        self._check_version()
        self._init_chunk_state()

    def _validate_json_file(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        if self.json_path.suffix != '.json':
            raise ValueError("File must be JSON format")

    def _check_version(self):
        version = self.data.get("version", None)
        if version is None or version != 1.0:
            raise ValueError(f"Unsupported or missing JSON version: {version}")

    # def _init_chunk_state(self):
    #     self.chunks = self.data.get("chunks", [])
    #     self.summary = self.data.get("summary", {})
    #     self._processed_set = set(self.summary.get("processed_ids", []))

    def _init_chunk_state(self):
        self.chunks = self.data.get("chunks", [])
        self.summary = self.data.get("summary", {})
        raw_ids = self.summary.get("processed_ids", [])
        self._processed_set = set(str(i) for i in raw_ids)

    @property
    def total_chunks(self) -> int:
        return self.summary.get("total_chunks", len(self.chunks))

    @property
    def remaining_chunks(self) -> int:
        return len(self._get_unprocessed_chunks())

    # def _get_unprocessed_chunks(self) -> List[Dict[str, Any]]:
    #     return [
    #         chunk for i, chunk in enumerate(self.chunks)
    #         if i not in self._processed_set
    #     ]

    def _get_unprocessed_chunks(self) -> List[Dict[str, Any]]:
        return [
            chunk for chunk in self.chunks
            if chunk.get("chunk_id") not in self._processed_set
        ]

    def get_next_chunk(self) -> Optional[pd.DataFrame]:
        """Returns the next unprocessed chunk as a DataFrame."""
        for i, chunk in enumerate(self.chunks):
            if i not in self._processed_set:
                self._current_index = i  # track for marking later
                return pd.DataFrame(chunk["data"])
        return None  # All processed

    def mark_chunk_processed(self):
        """Marks the last fetched chunk as processed."""
        if hasattr(self, "_current_index"):
            self._processed_set.add(self._current_index)
            self.summary["processed"] = len(self._processed_set)
            self.summary["processed_ids"] = sorted(self._processed_set)
            self._save_state()
            del self._current_index
        else:
            raise RuntimeError("No chunk fetched to mark as processed.")

    def _save_state(self):
        # Persist updated summary
        self.data["summary"] = self.summary
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def process_chunks(
            self,
            func: Callable[[pd.DataFrame], Any],
            show_progress: bool = True
    ) -> List[Any]:
        """
        Processes all unprocessed chunks with the provided function.

        Args:
            func: Function that takes a DataFrame and returns a result.
            show_progress: Whether to display a progress bar.

        Returns:
            A list of results from processing each chunk.
        """
        results = []
        unprocessed_chunks = self._get_unprocessed_chunks()

        iterator = enumerate(unprocessed_chunks)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Processing {self.json_path.name}",
                unit="chunk",
                total=len(unprocessed_chunks)
            )

        for _, chunk in iterator:
            try:
                df = pd.DataFrame(chunk["data"])
                result = func(df)
                results.append(result)

                chunk_id = chunk.get("chunk_id")
                if chunk_id:
                    self._processed_set.add(chunk_id)
            except Exception as e:
                results.append(f"Error: {str(e)}")

        # Save updated summary once at the end
        self.summary["processed"] = len(self._processed_set)
        self.summary["processed_ids"] = sorted(self._processed_set)
        self._save_state()

        return results

    # def process_chunks(
    #     self,
    #     func: Callable[[pd.DataFrame], Any],
    #     show_progress: bool = True
    # ) -> List[Any]:
    #     """Processes all unprocessed chunks sequentially."""
    #     results = []
    #     unprocessed_chunks = self._get_unprocessed_chunks()
    #
    #     iterator = enumerate(unprocessed_chunks)
    #     if show_progress:
    #         iterator = tqdm(
    #             iterator,
    #             desc=f"Processing {self.json_path.name}",
    #             unit="chunk",
    #             total=self.remaining_chunks
    #         )
    #
    #     for i, chunk in iterator:
    #         try:
    #             df = pd.DataFrame(chunk["data"])
    #             result = func(df)
    #             results.append(result)
    #             # Mark using global index, not i from the filtered list
    #             self._current_index = self.chunks.index(chunk)
    #             self.mark_chunk_processed()
    #         except Exception as e:
    #             results.append(f"Error: {str(e)}")
    #     return results

    def __repr__(self) -> str:
        return (
            f"ChunkManager(file='{self.json_path.name}', "
            f"processed={len(self._processed_set)}/{self.total_chunks})"
        )
