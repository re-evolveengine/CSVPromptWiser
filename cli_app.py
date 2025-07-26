from model.core.chunker import DataFrameChunker
from model.utils.chunk_json_inspector import ChunkJSONInspector
from model.utils.constants import TEMP_DIR
from model.utils.dataset_loader import DatasetLoader

inspector = ChunkJSONInspector()

def main():

    existing_file = inspector.find_valid_chunk_file()
    if existing_file:
        summary = inspector.inspect_chunk_file(existing_file)
        print(f"📁 Existing chunk file found: {summary['file']}")
        print(f"✅ Chunks: {summary['total_chunks']} | "
              f"Processed: {summary['processed_chunks']} | "
              f"Unprocessed: {summary['unprocessed_chunks']}")
        if summary["can_resume"]:
            print("➡️  You can resume processing from this file.")
        else:
            print("✔️ All chunks have been processed.")
    else:
        print("ℹ️ No valid chunked dataset found. You'll need to provide a dataset file.")


    # loader = DatasetLoader()
    # df = loader.load('profiles.csv')
    # chunker = DataFrameChunker()
    # chunks = chunker.chunk_dataframe(df, chunk_size=4)
    # chunker.save_chunks_to_json(chunks)


if __name__ == "__main__":
    main()
