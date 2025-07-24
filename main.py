import argparse
import pandas as pd

from model.core.chunk_manager import ChunkManager
# from model.core.chunker import DataFrameChunker
# from model.utils.data_prompt_arger import DataPromptArger
# from model.utils.dataset_loader import DatasetLoader


def dummy_process(df: pd.DataFrame) -> str:
    """Example processing function for chunks."""
    return f"Processed {len(df)} rows with columns: {list(df.columns)}"


def main():
    # --- CHUNKING PHASE (COMMENTED OUT) ---
    # chunker = DataFrameChunker(chunk_size=5000)
    # df = DatasetLoader().load(file_name="profiles.csv")
    # print(df.head(5))
    #
    # chunks = chunker.chunk_dataframe(df, 500)
    # print(f"Total Chunks: {len(chunks)}")
    #
    # DataFrameChunker.save_chunks_to_json(
    #     chunks,
    #     "C:\\Users\\Alchemist\\Documents\\Data\\profiles.json",
    #     max_rows_per_chunk=1000,
    #     metadata={"source": "production"}
    # )

    # --- CHUNK PROCESSING PHASE ---
    json_path = "C:\\Users\\Alchemist\\Documents\\Data\\profiles.json"
    manager = ChunkManager(json_path)

    print(f"\nLoaded {manager.total_chunks} chunks.")
    print(f"Remaining chunks to process: {manager.remaining_chunks}")

    results = manager.process_chunks(func=dummy_process)

    print("\nResults:")
    for i, result in enumerate(results):
        print(f"Chunk {i+1}: {result}")


if __name__ == "__main__":
    main()
