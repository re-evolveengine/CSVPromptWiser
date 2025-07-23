import argparse

from model.core.chucker import DataFrameChunker
from model.utils.data_prompt_arger import DataPromptArger
from model.utils.dataset_loader import DatasetLoader
import pandas as pd


def main():
    # Chunking
    chunker = DataFrameChunker(chunk_size=5000)
    df = DatasetLoader().load(file_name="profiles.csv")
    print(df.head(5))

    chunks = chunker.chunk_dataframe(df,500)
    print(f"Total Chunks: {len(chunks)}")

    # Saving
    DataFrameChunker.save_chunks_to_json(
        chunks,
        "C:\\Users\\Alchemist\\Documents\\Data\\profiles.json",
        max_rows_per_chunk=1000,
        metadata={"source": "production"}
    )

    # arger = DataPromptArger()
    #
    # try:
    #     print(f"\nPrompt: {arger.get_prompt()}")
    #     arger.print_df_head()
    #     arger.print_df_shape()
    # except Exception as e:
    #     print(f"Error: {e}")
    #     exit(1)


if __name__ == "__main__":
    main()