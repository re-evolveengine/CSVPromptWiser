# cli_utils.py

import os
import dotenv

from model.core.gemini_client import GeminiClient
from model.core.runners.gemini_resilient_runner import GeminiResilientRunner
import pandas as pd


from model.core.gemini_model_provider import GeminiModelProvider
from model.utils.model_prefs import ModelPreference


def handle_model_selection(api_key: str) -> str:
    """Handles model retrieval, selection, and persistence."""
    model_pref = ModelPreference()
    saved_model = model_pref.get_model_name()

    if saved_model:
        print(f"\n📌 A previously selected model was found: {saved_model}")
        use_saved = input("Do you want to use the saved model? [Y/n]: ").strip().lower()
        if use_saved in ['', 'y', 'yes']:
            print(f"\n✅ Using saved model: {saved_model}")
            return saved_model

    # No saved model or user chose not to use it
    print("\nFetching available Gemini models...\n")
    provider = GeminiModelProvider(api_key)
    model_names = provider.get_usable_model_names()

    if not model_names:
        print("❌ No usable models found. Please check your API key or network.")
        exit(1)

    print("Available Gemini Models:")
    for idx, name in enumerate(model_names, 1):
        print(f"{idx}. {name}")

    selected_model = get_model_selection(model_names)
    model_pref.save_model_name(selected_model)

    print(f"\n✅ Selected and saved model: {selected_model}")
    return selected_model



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


# def run_gemini_chunk_processor(prompt: str, model_name: str, api_key: str, chunk_manager) -> list:
#     """Run Gemini LLM over each chunk using resilient retry logic."""
#     client = GeminiClient(model=model_name, api_key=api_key)
#     runner = GeminiResilientRunner(client=client)
#
#     def process_fn(df: pd.DataFrame):
#         result = runner.run(prompt, df)
#         print(f"[Prompt] Applied to chunk of shape {df.shape}")
#         return result
#
#     return chunk_manager.process_chunks(process_fn)


def run_gemini_chunk_processor(prompt: str, model_name: str, api_key: str, chunk_manager) -> list:
    """Run Gemini LLM over each chunk using resilient retry logic."""
    client = GeminiClient(model=model_name, api_key=api_key)
    runner = GeminiResilientRunner(client=client)

    def process_fn(df: pd.DataFrame):
        response = runner.run(prompt, df)
        print(f"[Prompt] Applied to chunk of shape {df.shape}")
        return {
            "chunk": df,
            "prompt": prompt,
            "response": response
        }

    return chunk_manager.process_chunks(process_fn)

