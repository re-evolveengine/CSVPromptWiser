import json
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
from pathlib import Path

from model.utils.constants import JSON_CHUNK_VERSION


class ChunkManager:
    """Manages loading and tracking of chunks from JSON storage."""

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self._validate_json_file()
        self._current_chunk_id = None

        with open(self.json_path, "r") as f:
            self.data = json.load(f)

        self._check_version()
        self._init_chunk_state()


    @property
    def current_chunk_id(self):
        return self._current_chunk_id

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

    def get_next_chunk(self) -> Tuple[pd.DataFrame, Optional[str]]:
        """Returns the next unprocessed chunk as a DataFrame."""
        for chunk in self.chunks:
            chunk_id = str(chunk.get("chunk_id"))
            if chunk_id and chunk_id not in self._processed_set:
                self._current_chunk_id = chunk_id
                return pd.DataFrame(chunk["data"]), chunk_id
        return None

    def mark_chunk_processed(self, chunk_id: Optional[str] = None):
        """Mark the most recent or specified chunk as processed."""
        if chunk_id:
            self._processed_set.add(str(chunk_id))
        elif hasattr(self, "_current_chunk_id"):
            self._processed_set.add(self._current_chunk_id)
            self._current_chunk_id = None
        else:
            raise RuntimeError("No chunk to mark as processed.")

    def save_state(self):
        """Save updated processed chunk IDs to the JSON file."""
        self.summary["processed"] = len(self._processed_set)
        self.summary["processed_ids"] = sorted(self._processed_set)
        self.data["summary"] = self.summary

        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def __repr__(self) -> str:
        return (
            f"ChunkManager(file='{self.json_path.name}', "
            f"processed={len(self._processed_set)}/{self.total_chunks})"
        )
