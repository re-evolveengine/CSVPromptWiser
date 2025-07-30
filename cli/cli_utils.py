# cli_utils.py

import os
import dotenv
from click import Tuple
from factory import List
from tenacity import RetryError

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner
import pandas as pd


from model.core.llms.gemini_model_provider import GeminiModelProvider
from model.io.model_prefs import ModelPreference
from model.utils.constants import MODEL_PREFS_DB_PATH_CLI


def handle_model_selection(api_key: str) -> str:
    """Handles model retrieval, selection, and persistence."""
    model_pref = ModelPreference(db_path=MODEL_PREFS_DB_PATH_CLI)
    saved_models = model_pref.get_model_list()
    saved_selected_model = model_pref.get_selected_model_name()

    # 1. Check if a previously selected model exists
    if saved_selected_model:
        print(f"\n📌 A previously selected model was found: {saved_selected_model}")
        use_saved = input("Do you want to use the saved model? [Y/n]: ").strip().lower()
        if use_saved in ['', 'y', 'yes']:
            print(f"\n✅ Using saved model: {saved_selected_model}")
            return saved_selected_model

    # 2. Show saved model list, if available
    if saved_models:
        print("\n📌 Previously fetched models:")
        for idx, name in enumerate(saved_models, 1):
            print(f"{idx}. {name}")

        result = input("Do you want to use the saved models? [Y/n]: ").strip().lower()
        if result in ['', 'y', 'yes']:
            print(f"\n✅ Using saved models.")
            selected_model = get_model_selection(saved_models)
            model_pref.save_selected_model_name(selected_model)
            return selected_model

    # 3. User chose not to use saved models OR no saved models exist → fetch new ones
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

    # Persist selected model and full model list
    model_pref.save_selected_model_name(selected_model)
    model_pref.save_model_list(model_names)

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



def run_gemini_chunk_processor(
    prompt: str,
    model_name: str,
    api_key: str,
    chunk_manager,
    max_chunks: int = None
):
    """
    Applies the prompt to chunks via Gemini, respecting max_chunks and handling errors.

    Args:
        prompt (str): Prompt text to apply.
        model_name (str): Gemini model name.
        api_key (str): Gemini API key.
        chunk_manager: Instance of ChunkManager.
        max_chunks (int, optional): Max number of chunks to process.

    Returns:
        (results: list, any_success: bool)
    """
    client = GeminiClient(model=model_name, api_key=api_key)
    runner = GeminiResilientRunner(client=client)
    results = []
    any_success = False

    def process_fn(df: pd.DataFrame):
        nonlocal any_success
        try:
            response = runner.run(prompt, df)
            print(f"[Prompt] Applied to chunk of shape {df.shape}")
            results.append({
                "chunk": df,
                "prompt": prompt,
                "response": response
            })
            any_success = True
        except runner.user_errors as ue:
            print(f"[User Error] Skipping chunk due to user error: {ue}")
        except RetryError as re:
            last_exc = re.last_attempt.exception()
            print(f"[Retryable Error] Skipping chunk after max retries. Last error: {last_exc}")
        except Exception as e:
            print(f"[Unexpected Error] Skipping chunk due to unknown error: {e}")

    try:
        chunk_manager.process_chunks(
            func=process_fn,
            show_progress=True,
            max_chunks=max_chunks  # ✅ This was missing
        )
    except Exception as e:
        print(f"[Fatal Error] Something went wrong during chunk processing: {e}")
        return results, False

    return results, any_success





