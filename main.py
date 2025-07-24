import os
import dotenv
import pandas as pd

from model.core.chunker import DataFrameChunker
from model.core.chunk_manager import ChunkManager
from model.core.gemini_client import GeminiClient
from model.utils.dataset_loader import DatasetLoader

# Load environment variables
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

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



def main():
    json_path = "C:\\Users\\Alchemist\\Documents\\Data\\profiles.json"

    loader = DatasetLoader()
    df = loader.load("profiles.csv")

    chunker = DataFrameChunker(chunk_size=4)
    data_frame_chunks = chunker.chunk_dataframe(df)
    chunker.save_chunks_to_json(chunks=data_frame_chunks, file_path=json_path)

    manager = ChunkManager(json_path)
    print(f"\nLoaded {manager.total_chunks} chunks.")
    print(f"Remaining chunks to process: {manager.remaining_chunks}")
    first_chunk = manager.get_next_chunk()
    print(f"\nFirst chunk:\n{first_chunk}")

    results = manager.process_chunks(func=process_with_gemini,show_progress=True,max_chunks=1)

    manager.mark_chunk_processed()

    print(f"\nRemaining chunks to process: {manager.remaining_chunks}")

    print("\nResults:")
    for i, result in enumerate(results):
        print(f"\n--- Chunk {i + 1} ---\n{result}")


if __name__ == "__main__":
    main()
