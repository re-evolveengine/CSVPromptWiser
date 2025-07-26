# cli_utils.py

import os
import dotenv

from model.core.gemini_client import GeminiClient
from model.core.runners.gemini_resilient_runner import GeminiResilientRunner
import pandas as pd


def load_api_key() -> str:
    """Load API key from .env or ask user."""
    dotenv.load_dotenv()
    api_key = os.getenv("KEY")

    if not api_key:
        api_key = input("Enter your Gemini API key: ").strip()
        dotenv.set_key(dotenv.find_dotenv(), "KEY", api_key)
        print("✅ API key saved to .env.")
    else:
        print("✅ API key loaded from .env.")

    return api_key


def get_model_selection(model_names: list[str]) -> str:
    """Prompt user to select model by number."""
    while True:
        try:
            selection = int(input("\nChoose a model by number: "))
            if 1 <= selection <= len(model_names):
                return model_names[selection - 1]
            print("Invalid number. Please choose from the list.")
        except ValueError:
            print("Please enter a valid integer.")


def ask_int_input(msg: str) -> int:
    """Generic integer input helper."""
    while True:
        try:
            return int(input(msg))
        except ValueError:
            print("Please enter a valid integer.")


def run_gemini_chunk_processor(prompt: str, model_name: str, api_key: str, chunk_manager) -> list:
    """Run Gemini LLM over each chunk using resilient retry logic."""
    client = GeminiClient(model=model_name, api_key=api_key)
    runner = GeminiResilientRunner(client=client)

    def process_fn(df: pd.DataFrame):
        result = runner.run(prompt, df)
        print(f"[Prompt] Applied to chunk of shape {df.shape}")
        return result

    return chunk_manager.process_chunks(process_fn)
