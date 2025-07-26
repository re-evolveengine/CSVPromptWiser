import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from model.utils.constants import TEMP_DIR


class ChunkJSONInspector:
    """
    Inspects JSON files in the temp directory for chunking structure,
    especially to detect resumable processing state.
    """

    def __init__(self, directory_path: str = TEMP_DIR):
        os.makedirs(directory_path, exist_ok=True)

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        self.directory = Path(directory_path)

    def find_valid_chunk_file(self) -> Optional[Path]:
        """
        Scans the directory for a JSON file with expected structure.

        Returns:
            Path to valid chunk JSON file or None if not found.
        """
        for file in self.directory.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if self._is_valid_chunk_json(data):
                        return file
            except Exception:
                continue
        return None

    def inspect_chunk_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parses and summarizes the chunk file.

        Args:
            file_path: Path to chunk JSON

        Returns:
            Dictionary with total_chunks, processed_ids, and unprocessed_count
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        total = summary.get("total_chunks", 0)
        processed_ids = set(summary.get("processed_ids", []))

        chunks = data.get("chunks", [])
        all_chunk_ids = {chunk.get("chunk_id") for chunk in chunks if "chunk_id" in chunk}
        unprocessed = all_chunk_ids - processed_ids

        return {
            "file": str(file_path.name),
            "version": data.get("version", "unknown"),
            "total_chunks": total,
            "processed_chunks": len(processed_ids),
            "unprocessed_chunks": len(unprocessed),
            "can_resume": len(unprocessed) > 0
        }

    @staticmethod
    def _is_valid_chunk_json(data: Dict[str, Any]) -> bool:
        """
        Lightweight structural validation.

        Args:
            data: Loaded JSON

        Returns:
            True if structure looks like a valid chunk file
        """
        return (
            isinstance(data, dict)
            and isinstance(data.get("chunks"), list)
            and isinstance(data.get("summary"), dict)
            and "total_chunks" in data["summary"]
            and all("chunk_id" in chunk and "data" in chunk for chunk in data["chunks"])
        )
