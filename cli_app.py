from model.core.chunker import DataFrameChunker
from model.utils.chunk_json_inspector import ChunkJSONInspector
from model.utils.constants import TEMP_DIR
from model.utils.data_prompt_arger import DataPromptArger
from model.utils.dataset_loader import DatasetLoader


def main():
    arger = DataPromptArger()
    inspector = ChunkJSONInspector()
    command = arger.get_command()

    if command == "resume":
        inspector = ChunkJSONInspector()
        existing_file = inspector.find_valid_chunk_file()

        if existing_file:
            summary = inspector.inspect_chunk_file(existing_file)
            print(f"📁 Existing chunk file found: {summary['file']}")
            print(f"✅ Chunks: {summary['total_chunks']} | "
                  f"Processed: {summary['processed_chunks']} | "
                  f"Unprocessed: {summary['unprocessed_chunks']}")
            if summary["can_resume"]:
                print("➡️  Resuming is possible.")
                # You could trigger the resume pipeline here
            else:
                print("✔️ All chunks are already processed.")
        else:
            print("⚠️ No valid chunked dataset to resume from.")

    elif command == "new":
        df = arger.get_dataset()
        prompt = arger.get_prompt()
        chunk_size = arger.get_chunk_size()

        arger.print_df_shape()
        arger.print_df_head()

        chunker = DataFrameChunker(chunk_size=chunk_size)
        chunks = chunker.chunk_dataframe(df)

        out_path = TEMP_DIR / "chunks.json"
        chunker.save_chunks_to_json(chunks, str(out_path), metadata={"prompt": prompt})
        print(f"✅ New chunks saved to: {out_path}")

    # loader = DatasetLoader()
    # df = loader.load('profiles.csv')
    # chunker = DataFrameChunker()
    # chunks = chunker.chunk_dataframe(df, chunk_size=4)
    # chunker.save_chunks_to_json(chunks)


if __name__ == "__main__":
    main()
