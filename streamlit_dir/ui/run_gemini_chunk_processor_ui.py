import pandas as pd
from tenacity import RetryError
from typing import List, Tuple, Optional, Callable, Generator

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner
from model.io.model_prefs import ModelPreference


# def run_gemini_chunk_processor_ui(
#     prompt: str,
#     client: GeminiClient,
#     chunk_manager: ChunkManager,
#     max_chunks: Optional[int] = None,
#     progress_callback: Optional[Callable[[int, int, int], None]] = None
# ) -> Tuple[List[dict], List[str]]:
#     """
#     Applies the given prompt to up to `max_chunks` data chunks using a pre-initialized Gemini client.
#
#     Args:
#         prompt (str): The user prompt.
#         client (BaseLLMClient): An already configured GeminiClient instance.
#         chunk_manager (ChunkManager): Manages access to the chunked DataFrame.
#         max_chunks (int, optional): Maximum number of chunks to process.
#         progress_callback (callable, optional): Function to call with (i, total) after each chunk is processed.
#
#     Returns:
#         results: List of dicts, each with keys {"chunk", "prompt", "response"}.
#         errors: List of any chunk-level error messages encountered.
#     """
#     runner = GeminiResilientRunner(client=client)
#     results: List[dict] = []
#     errors: List[str] = []
#
#     if client is None or prompt is None or chunk_manager is None:
#         errors.append("[Fatal Error] Client, prompt, or chunk manager is None.")
#         return results, errors
#
#     total = chunk_manager.remaining_chunks if max_chunks is None else min(max_chunks, chunk_manager.remaining_chunks)
#     prefs = ModelPreference()
#     count = 0
#     current_remaining_total = prefs.get_remaining_total_tokens()
#
#     def process_fn(df: pd.DataFrame):
#         nonlocal count, current_remaining_total
#         try:
#             response, used_tokens = runner.run(prompt, df)
#
#             # Persist token usage to disk immediately
#             current_remaining_total -= used_tokens
#             prefs.save_remaining_total_tokens(current_remaining_total)
#
#             # Save result
#             results.append({
#                 "chunk": df,
#                 "prompt": prompt,
#                 "response": response,
#                 "remaining_tokens": current_remaining_total
#             })
#
#         except runner.user_errors as ue:
#             errors.append(f"[User Error] Skipped chunk: {ue}")
#         except RetryError as re:
#             last_exc = re.last_attempt.exception()
#             errors.append(f"[Retryable Error] Skipped chunk after retries: {last_exc}")
#         except Exception as e:
#             errors.append(f"[Unexpected Error] Skipped chunk: {e}")
#         finally:
#             count += 1
#             if progress_callback:
#                 progress_callback(count, total, current_remaining_total)
#
#     try:
#         chunk_manager.process_chunks(
#             func=process_fn,
#             show_progress=False,
#             max_chunks=max_chunks
#         )
#     except Exception as e:
#         errors.append(f"[Fatal Error] Processing aborted: {e}")
#
#     return results, errors
#
#
#
#









def process_next_chunk_generator(
    prompt: str,
    client: GeminiClient,
    chunk_manager: ChunkManager,
    chunk_count: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None
) -> Generator[dict, None, None]:
    """
    Generator that processes one chunk at a time using Gemini and yields results incrementally.

    Yields:
        dict: A structured result dict:
              - On success: { "status": "ok", "data": ..., "remaining_tokens": ... }
              - On failure: { "status": "error", "error": ..., "remaining_tokens": ... }
    """
    runner = GeminiResilientRunner(client=client)
    prefs = ModelPreference()

    total = chunk_manager.remaining_chunks if chunk_count is None else min(chunk_count, chunk_manager.remaining_chunks)
    count = 0
    current_remaining_total = prefs.get_remaining_total_tokens()

    chunks_processed = 0
    for chunk_id, chunk_df in chunk_manager.get_chunk_iterator(max_chunks=chunk_count):
        error = None
        result = None

        try:
            response, used_tokens = runner.run(prompt, chunk_df)

            # Update token count
            current_remaining_total -= used_tokens
            prefs.save_remaining_total_tokens(current_remaining_total)

            # Build result
            result = {
                "chunk": chunk_df,
                "prompt": prompt,
                "response": response,
                "remaining_tokens": current_remaining_total
            }

            chunk_manager.mark_chunk_processed(chunk_id)

        except runner.user_errors as ue:
            error = f"[User Error] Skipped chunk: {ue}"
        except RetryError as re:
            last_exc = re.last_attempt.exception()
            error = f"[Retryable Error] Skipped chunk after retries: {last_exc}"
        except Exception as e:
            error = f"[Unexpected Error] Skipped chunk: {e}"

        # Update count and progress
        count += 1
        if progress_callback:
            progress_callback(count, total, current_remaining_total)

        # Yield structured result
        if error:
            yield {
                "status": "error",
                "error": error,
                "remaining_tokens": current_remaining_total
            }
        else:
            yield {
                "status": "ok",
                "data": result,
                "remaining_tokens": current_remaining_total
            }

        # Optional: break early if max_chunks is reached
        chunks_processed += 1
        if chunk_count and chunks_processed >= chunk_count:
            break

