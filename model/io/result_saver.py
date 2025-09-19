import json
import os
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime
import pandas as pd

from utils.constants import JSON_CHUNK_VERSION


class ResultSaver:
    """Handles saving Gemini results along with input chunks to JSON."""

    @staticmethod
    def save_results_to_json(
        results: List[Dict[str, Any]],
        file_path: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Save Gemini responses with input and prompt to JSON.

        Args:
            results: List of dicts each containing:
                     - 'chunk': Union[pd.DataFrame, List[Dict]] - Input data as DataFrame or list of dicts
                     - 'prompt': str - The prompt used for this chunk
                     - 'response': str - The model's response
            file_path: Destination JSON file path
            metadata: Optional additional metadata

        Raises:
            ValueError: If no results provided or invalid chunk format
            OSError: If write fails
        """
        if not results:
            raise ValueError("No results to save")

        output = {
            "version": JSON_CHUNK_VERSION,
            "metadata": metadata or {},
            "chunks": [],
            "summary": {
                "total_chunks": len(results),
                "processed_ids": []
            }
        }

        for item in results:
            chunk = item["chunk"]
            prompt = item["prompt"]
            response = item["response"]

            # Handle both DataFrame and list of dicts for chunk
            if isinstance(chunk, pd.DataFrame):
                chunk_data = chunk.to_dict(orient="records")
                original_rows = len(chunk)
            elif isinstance(chunk, list) and all(isinstance(x, dict) for x in chunk):
                chunk_data = chunk
                original_rows = len(chunk)
            else:
                raise ValueError(
                    "Chunk must be a pandas DataFrame or a list of dictionaries. "
                    f"Got {type(chunk).__name__} instead."
                )

            chunk_id = str(uuid4())
            output["chunks"].append({
                "chunk_id": chunk_id,
                "data": chunk_data,
                "original_rows": original_rows,
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            output["summary"]["processed_ids"].append(chunk_id)

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")

        try:
            with open(temp_path, 'w') as f:
                json.dump(output, f, indent=2)
            os.replace(temp_path, path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise OSError(f"Failed to save results: {str(e)}") from e

        print(f"Saved {len(results)} Gemini results to {file_path}")



    @staticmethod
    def save_results_to_csv(
        results: List[Dict[str, Any]],
        file_path: str
    ) -> None:
        """
        Save Gemini responses with input and prompt to a flat CSV where each row corresponds
        to an input row from a chunk, enriched with metadata and model response.

        Args:
            results: List of dicts each containing:
                     - 'chunk': pd.DataFrame or list of dicts
                     - 'prompt': str
                     - 'response': str
            file_path: Destination CSV path

        Raises:
            ValueError: If results are empty or invalid
        """
        if not results:
            raise ValueError("No results to save")

        rows = []

        for i, item in enumerate(results):
            chunk = item["chunk"]
            prompt = item["prompt"]
            response = item["response"]
            timestamp = item.get("timestamp") or datetime.utcnow().isoformat() + "Z"
            chunk_id = item.get("chunk_id") or str(uuid4())

            if isinstance(chunk, pd.DataFrame):
                chunk_data = chunk.to_dict(orient="records")
            elif isinstance(chunk, list) and all(isinstance(x, dict) for x in chunk):
                chunk_data = chunk
            else:
                raise ValueError(f"Invalid chunk format at index {i}")

            for row in chunk_data:
                enriched_row = {
                    **row,  # original input fields
                    "chunk_id": chunk_id,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": timestamp
                }
                rows.append(enriched_row)

        df = pd.DataFrame(rows)
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=False)
        print(f"Saved {len(rows)} rows (from {len(results)} chunks) to {file_path}")
