import os
import dotenv
import pandas as pd

from model.core.chunker import DataFrameChunker
from model.core.chunk_manager import ChunkManager
from model.core.error_handler.gemini_error_handler import GeminiErrorHandler
from model.core.gemini_client import GeminiClient
from model.core.gemini_model_provider import GeminiModelProvider
from model.core.runners.gemini_resilient_runner import GeminiResilientRunner
from model.utils.dataset_loader import DatasetLoader

# Load environment variables
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")


# import threading
#
# def listen_for_pause(manager):
#     while True:
#         cmd = input("Press [p] to pause/resume: ").strip().lower()
#         if cmd == "p":
#             if manager.is_paused():
#                 print("▶️  Resuming...")
#                 manager.resume()
#             else:
#                 print("⏸️  Pausing...")
#                 manager.pause()
#
# # Usage in your CLI entrypoint:
# chunk_manager = ChunkManager("path/to/your.json")
# threading.Thread(target=listen_for_pause, args=(chunk_manager,), daemon=True).start()
# chunk_manager.process_chunks(my_processing_function)


def process_with_gemini(df: pd.DataFrame) -> str:
    """
    Sends chunk to Gemini for summarization or analysis.
    """
    prompt = (
        "You are a data analyst. Summarize key trends, patterns, or anomalies "
        "in this dataset chunk in plain English."
    )

    client = GeminiClient(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)
    return client.call(prompt, df)

# def process_chunk_with_prompt(df):
#     try:
#         result = runner.run(prompt, df)
#         print("[✅ Success]")
#         return result
#     except runner.user_errors as e:
#         print(f"[❌ User Error] {type(e).__name__}: {e}")
#         return f"User Error: {str(e)}"
#     except runner.retryable_errors as e:
#         print(f"[🔁 Retryable Error] {type(e).__name__}: {e}")
#         return f"Retryable Error: {str(e)}"
#     except Exception as e:
#         print(f"[❓ Unknown Error] {type(e).__name__}: {e}")
#         return f"Unknown Error: {str(e)}"


# def main():
#     json_path = "C:\\Users\\Alchemist\\Documents\\Data\\profiles.json"
#
#     loader = DatasetLoader()
#     df = loader.load("profiles.csv")
#
#     chunker = DataFrameChunker(chunk_size=4)
#     data_frame_chunks = chunker.chunk_dataframe(df)
#     chunker.save_chunks_to_json(chunks=data_frame_chunks, file_path=json_path)
#
#     manager = ChunkManager(json_path)
#     print(f"\nLoaded {manager.total_chunks} chunks.")
#     print(f"Remaining chunks to process: {manager.remaining_chunks}")
#     first_chunk = manager.get_next_chunk()
#     print(f"\nFirst chunk:\n{first_chunk}")
#
#     results = manager.process_chunks(func=process_with_gemini,show_progress=True,max_chunks=1)
#
#     manager.mark_chunk_processed()
#
#     print(f"\nRemaining chunks to process: {manager.remaining_chunks}")
#
#     print("\nResults:")
#     for i, result in enumerate(results):
#         print(f"\n--- Chunk {i + 1} ---\n{result}")


def main():
    import pandas as pd
    import os
    from model.core.gemini_client import GeminiClient
    from model.core.runners.gemini_resilient_runner import GeminiResilientRunner

    prompt = "Summarize the sales performance in this table:"

    df = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "Units Sold": [100, 150, 80],
        "Revenue": [1000, 2300, 900]
    })

    api_key = os.getenv("KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")

    client = GeminiClient(api_key=api_key, model="models/gemini-1.5-flash")
    runner = GeminiResilientRunner(client)

    response = runner.run(prompt, df)
    print(response)


def manager



if __name__ == "__main__":
    main()
