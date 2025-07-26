import os
import threading
import time
from datetime import datetime
from pathlib import Path

import keyboard
import pandas as pd

from model.core.chunk_manager import ChunkManager
from model.core.chunker import DataFrameChunker
from model.core.gemini_model_provider import GeminiModelProvider
from model.utils.cli_utils import load_api_key, get_model_selection, ask_int_input, run_gemini_chunk_processor, \
    handle_model_selection
from model.utils.constants import TEMP_DIR, DATA_DIR, RESULTS_DIR
from model.utils.chunk_json_inspector import ChunkJSONInspector
from model.utils.dataset_loader import DatasetLoader
from model.utils.gemini_result_saver import GeminiResultSaver


class CLIFlowController:
    def __init__(self):
        self.chunk_file = None
        self.prompt = None
        self.num_chunks = None
        self.resume = False
        self.running = True
        self.paused = False
        self.results = None

        self.original_df = None

        self.loader = DatasetLoader()
        self.chunker = DataFrameChunker()
        self.chunk_manager = None
        self.api_key = None
        self.model_name = None

    def run(self):
        self.step_0_choose_model()
        self.step_1_check_existing_chunk_file()

        if not self.resume:
            self.step_2_ask_for_dataset()
            self.step_3_chunk_dataframe()

        self.step_4_process_chunks()

    def step_0_choose_model(self):
        print("=== Step 0: Select a Gemini Model ===")
        self.api_key = load_api_key()
        self.model_name = handle_model_selection(self.api_key)

    def step_1_check_existing_chunk_file(self):
        inspector = ChunkJSONInspector(directory_path=TEMP_DIR)
        chunk_file = inspector.find_valid_chunk_file()

        if chunk_file:
            info = inspector.inspect_chunk_file(chunk_file)
            self.chunk_file = chunk_file
            print("\nExisting chunk file found:")
            print(f"  File: {info['file']}")
            print(f"  Total Chunks: {info['total_chunks']}")
            print(f"  Processed: {info['processed_chunks']}")
            print(f"  Unprocessed: {info['unprocessed_chunks']}")
            print(f"  Can Resume: {info['can_resume']}")
            print(f"  Chunk Size: {info['chunk_size']}")

            if info['can_resume'] > 0:
                answer = input("\nStart processing from where you left off? [y/N]: ").strip().lower()
                if answer == 'y':
                    self.resume = True

    def step_2_ask_for_dataset(self):
        print("\nLooking for dataset files in the data directory...")
        files = list(Path(DATA_DIR).glob("*.csv")) + list(Path(DATA_DIR).glob("*.parquet"))

        if not files:
            print("No dataset found. Please copy your .csv or .parquet file into the data directory:")
            print(f"  {DATA_DIR}")

        while True:
            filename = input("Enter dataset filename (e.g. sales.csv): ").strip()
            dataset_path = Path(DATA_DIR) / filename
            if dataset_path.exists():
                print(f"Found: {dataset_path.name}")
                break
            print("File not found. Please try again.")

        self.original_df = self.loader.load(filename)

    def step_3_chunk_dataframe(self):
        chunk_size = ask_int_input("\nReady to chunk dataset? Enter the chunk size: ")

        print("[Chunking started...]")
        chunks = self.chunker.chunk_dataframe(self.original_df, chunk_size)

        json_path = Path(TEMP_DIR) / "chunks.json"
        self.chunker.save_chunks_to_json(chunks=chunks, file_path=str(json_path))
        self.chunk_file = json_path

        print("[Chunking complete and saved to JSON]")

    def step_4_process_chunks(self):
        self.prompt = input("\nEnter the prompt to process each chunk: ").strip()
        self.num_chunks = ask_int_input("Enter number of chunks to process: ")

        self.chunk_manager = ChunkManager(json_path=str(self.chunk_file))

        self.paused = False
        self.running = True

        keyboard_thread = threading.Thread(target=self._keyboard_listener)
        keyboard_thread.start()

        print("\nProcessing... Press 'p' to pause, 'r' to resume, 'e' to exit")

        try:
            self.results = run_gemini_chunk_processor(
                prompt=self.prompt,
                model_name=self.model_name,
                api_key=self.api_key,
                chunk_manager=self.chunk_manager,
            )
            print("\nProcessing complete.")
        finally:
            self.running = False
            keyboard_thread.join()

            if self.results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"{self.model_name}_{timestamp}"
                json_path = Path(RESULTS_DIR) / f"{base_name}.json"
                csv_path = Path(RESULTS_DIR) / f"{base_name}.csv"

                GeminiResultSaver.save_results_to_json(self.results, str(json_path))
                GeminiResultSaver.save_results_to_csv(self.results, str(csv_path))
            else:
                print("⚠️ No results to save.")

    def _keyboard_listener(self):
        while self.running:
            if keyboard.is_pressed('p'):
                if not self.chunk_manager.is_paused():
                    self.chunk_manager.pause()
                    print("[Paused]")
            elif keyboard.is_pressed('r'):
                if self.chunk_manager.is_paused():
                    self.chunk_manager.resume()
                    print("[Resumed]")
            elif keyboard.is_pressed('e'):
                self.running = False
                self.chunk_manager.resume()  # unblock pause wait
                print("[Exiting...]")
                break
            time.sleep(0.1)
