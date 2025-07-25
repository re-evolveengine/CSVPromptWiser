import json
from typing import Any, Callable, Optional, Dict, List
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from threading import Event

from model.utils.constants import JSON_CHUNK_VERSION


class ChunkManager:
    """Manages sequential processing of chunks from JSON storage."""

    def __init__(self, json_path: str):
        """
        Args:
            json_path: Path to JSON file created by DataFrameChunker.save_chunks_to_json()
        """
        self.json_path = Path(json_path)
        self._validate_json_file()

        with open(self.json_path, "r") as f:
            self.data = json.load(f)

        self._check_version()
        self._init_chunk_state()

        # 🧵 Pause event for CLI control
        self.pause_event = Event()
        self.pause_event.set()  # Start in 'resumed' state

    def pause(self):
        """Pause the chunk processing."""
        self.pause_event.clear()

    def resume(self):
        """Resume the chunk processing."""
        self.pause_event.set()

    def is_paused(self) -> bool:
        return not self.pause_event.is_set()

    def _validate_json_file(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        if self.json_path.suffix != ".json":
            raise ValueError("File must be JSON format")

    def _check_version(self):
        version = self.data.get("version", None)
        if version is None or version != JSON_CHUNK_VERSION:
            raise ValueError(f"Unsupported or missing JSON version: {version}")

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

    def _get_unprocessed_chunks(self) -> List[Dict[str, Any]]:
        return [
            chunk for chunk in self.chunks
            if str(chunk.get("chunk_id")) not in self._processed_set
        ]

    def get_next_chunk(self) -> Optional[pd.DataFrame]:
        """Returns the next unprocessed chunk as a DataFrame."""
        for chunk in self.chunks:
            chunk_id = str(chunk.get("chunk_id"))
            if chunk_id and chunk_id not in self._processed_set:
                self._current_chunk_id = chunk_id
                return pd.DataFrame(chunk["data"])
        return None  # All processed

    def mark_chunk_processed(self):
        """Marks the most recently fetched chunk as processed."""
        if hasattr(self, "_current_chunk_id"):
            self._processed_set.add(self._current_chunk_id)
            self._current_chunk_id = None
        else:
            raise RuntimeError("No chunk fetched to mark as processed.")

    def _save_state(self):
        """Writes updated metadata back to the JSON file."""
        self.summary["processed"] = len(self._processed_set)
        self.summary["processed_ids"] = sorted(self._processed_set)
        self.data["summary"] = self.summary

        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def process_chunks(
            self,
            func: Callable[[pd.DataFrame], Any],
            show_progress: bool = True,
            max_chunks: Optional[int] = None
    ) -> List[Any]:
        """
        Process a specified number of unprocessed chunks sequentially.

        Args:
            func: Function that takes a DataFrame and returns any result.
            show_progress: Whether to show a progress bar.
            max_chunks: Optional number of chunks to process (None = all remaining).

        Returns:
            List of results from processing each chunk.
        """
        results = []
        unprocessed_chunks = self._get_unprocessed_chunks()

        if max_chunks is not None:
            unprocessed_chunks = unprocessed_chunks[:max_chunks]

        iterator = enumerate(unprocessed_chunks)

        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(unprocessed_chunks),
                desc=f"Processing {self.json_path.name}",
                unit="chunk"
            )

        for i, chunk in iterator:
            self.pause_event.wait()  # 🛑 Pause here if paused

            try:
                df = pd.DataFrame(chunk["data"])
                result = func(df)
                results.append(result)

                chunk_id = chunk.get("chunk_id")
                if chunk_id:
                    self._processed_set.add(chunk_id)

            except Exception as e:
                results.append(f"Error: {str(e)}")

        self.summary["processed"] = len(self._processed_set)
        self.summary["processed_ids"] = sorted(self._processed_set)
        self._save_state()

        return results

    def __repr__(self) -> str:
        return (
            f"ChunkManager(file='{self.json_path.name}', "
            f"processed={len(self._processed_set)}/{self.total_chunks})"
        )
