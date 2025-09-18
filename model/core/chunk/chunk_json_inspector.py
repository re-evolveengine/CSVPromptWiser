import json
from pathlib import Path
from typing import Optional, Dict, Any

from utils.constants import TEMP_DIR


class ChunkJSONInspector:
    """
    Inspects JSON files in the temp directory for chunking structure,
    especially to detect resumable processing state.
    """

    def __init__(self, directory_path: str = TEMP_DIR):
        """
        Initialize the ChunkJSONInspector.

        Args:
            directory_path: Path to the directory containing chunk JSON files.
                           If not provided, uses the default TEMP_DIR.

        Raises:
            FileNotFoundError: If the specified directory does not exist and its parent directory
                             doesn't exist either.
        """
        # Convert to Path object for easier manipulation
        dir_path = Path(directory_path).resolve()
        
        # If the directory doesn't exist
        if not dir_path.exists():
            # Check if parent directory exists
            if not dir_path.parent.exists():
                raise FileNotFoundError(
                    f"Parent directory does not exist: {dir_path.parent}"
                )
            # If parent exists, create the directory
            try:
                dir_path.mkdir(exist_ok=True)
            except OSError as e:
                raise FileNotFoundError(
                    f"Could not create directory {dir_path}: {e}"
                ) from e
        
        # If we get here, either the directory existed or was successfully created
        self.directory = dir_path

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
        chunk_size = summary.get("chunk_size", 0)
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
            "can_resume": len(unprocessed) > 0,
            "chunk_size": chunk_size,
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
