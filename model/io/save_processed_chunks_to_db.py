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
    if result.result_type != ResultType.SUCCESS:
        return

    if result.chunk is None:
        raise ValueError("Missing chunk in result for saving.")

    # Parse per-row responses
    response_lines = result.response.strip().splitlines()
    responses = [line.split(":", 1)[1].strip() for line in response_lines if ":" in line]

    if len(responses) != len(result.chunk):
        raise ValueError("Mismatch between response lines and chunk rows.")

    rows_to_save = []
    for i, (_, row) in enumerate(result.chunk.iterrows()):
        rows_to_save.append({
            "source_id": row["source_id"],
            "chunk_id": chunk_id,
            "prompt": prompt,
            "response": responses[i],
            "used_tokens": result.remaining_tokens,
            "model_version": model_version,
        })

    saver.save(rows_to_save)
