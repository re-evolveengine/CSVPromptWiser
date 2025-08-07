from typing import List
import pandas as pd

from model.io.gemini_sqlite_result_saver import GeminiSQLiteResultSaver
from model.utils.chunk_process_result import ChunkProcessResult
from model.utils.result_type import ResultType


def save_processed_chunk_to_db(
    result: ChunkProcessResult,
    chunk_id: str,
    prompt: str,
    model_version: str,
    saver: GeminiSQLiteResultSaver
) -> None:
    """
    Save a successfully processed chunk to the SQLite results database.

    Args:
        result (ChunkProcessResult): The result returned by GeminiChunkProcessor.
        chunk_id (str): The ID of the processed chunk.
        prompt (str): The prompt used for generation.
        model_version (str): The Gemini model version used.
        saver (GeminiSQLiteResultSaver): An instance of the DB saver.
    """
    if result.result_type != ResultType.SUCCESS:
        return  # only save successful results

    if result.chunk is None:
        raise ValueError("Missing chunk in result for saving.")

    rows_to_save = [
        {
            "source_id": row["source_id"],
            "chunk_id": chunk_id,
            "prompt": prompt,
            "response": result.response,
            "used_tokens": result.remaining_tokens,
            "model_version": model_version,
        }
        for _, row in result.chunk.iterrows()
    ]

    saver.save(rows_to_save)
