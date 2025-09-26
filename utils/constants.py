import os
from pathlib import Path

JSON_CHUNK_VERSION = 1.0
APP_NAME = "CSV PromptWiser"
DATA_FOLDER_NAME = "data"
RESULTS_FOLDER_NAME = "results"
RESULTS_DB_NAME = "processed_chunks.db"
TEMP_FOLDER_NAME = "temp"
CONFIG_FOLDER_NAME = "config"  # Optional: for preferences/configs

# 📂 Directories
DOCUMENTS_DIR = os.path.join(os.path.expanduser("~"), "Documents")
APP_DIR = os.path.join(DOCUMENTS_DIR, APP_NAME)
DATA_DIR = os.path.join(APP_DIR, DATA_FOLDER_NAME)
RESULTS_DIR = os.path.join(APP_DIR, RESULTS_FOLDER_NAME)
TEMP_DIR = os.path.join(APP_DIR, TEMP_FOLDER_NAME)
CONFIG_DIR = os.path.join(APP_DIR, CONFIG_FOLDER_NAME)
RESULTS_DB_PATH = os.path.join(RESULTS_DIR, RESULTS_DB_NAME)

# 📂 Files
JSON_CHUNK_FILE = os.path.join(TEMP_DIR, "chunks.json")

DEFAULT_CHUNK_SIZE = 25
DEFAULT_TOKEN_BUDGET = 10000

# 🗄️ Model preference DB (shelve) file name and full path
MODEL_PREFS_DB_NAME = "model_prefs.db"
MODEL_PREFS_DB_PATH = os.path.join(CONFIG_DIR, MODEL_PREFS_DB_NAME)
MODEL_KEY = "gemini_model"
MODEL_LIST_KEY = "gemini_model_list"
MODEL_CONFIG_KEY = "gemini_model_config"
REMAINING_TOTAL_TOKENS_KEY = "remaining_total_tokens"
TOTAL_TOKENS_KEY = "total_tokens_key"
CHUNK_SIZE_KEY = "chunk_size_key2"

# 📝 Prompt preferences file
PROMPT_PREF_PATH = Path(CONFIG_DIR) / ".prompt_pref.json"

# 🗄️ Model preference DB (shelve) file name and full path
MODEL_PREFS_DB_NAME = "model_prefs.db"


# Generation behavior
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 1.0

# Model token limits (maximum context window sizes)
MODEL_TOKEN_LIMITS = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-32k": 32768,
    "gemini-1.5-pro": 1048576,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-flash-8b": 1048576,
    "gemini-2.5-flash-preview-05-20": 1048576,
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-flash-lite-preview-06-17": 1048576,
    "gemini-2.5-pro": 1048576,
    "gemini-2.0-flash-exp": 1048576,
    "gemini-2.0-flash": 1048576,
    "gemini-2.0-flash-001": 1048576,
    "gemini-2.0-flash-lite-001": 1048576,
    "gemini-2.0-flash-lite": 1048576,
    "gemini-2.0-flash-lite-preview-02-05": 1048576,
    "gemini-2.0-flash-lite-preview": 1048576,
    "gemini-2.0-flash-thinking-exp-01-21": 1048576,
    "gemini-2.0-flash-thinking-exp": 1048576,
    "gemini-2.0-flash-thinking-exp-1219": 1048576,
    "learnlm-2.0-flash-experimental": 1048576,
    "gemma-3-1b-it": 8192,
    "gemma-3-4b-it": 8192,
    "gemma-3-12b-it": 8192,
    "gemma-3-27b-it": 8192,
    "gemma-3n-e4b-it": 8192,
    "gemma-3n-e2b-it": 8192,
    "gemini-2.5-flash-lite": 1048576,
}

# Safe prompt limits for practical chunking (much lower than max context window)
SAFE_PROMPT_LIMITS = {
    # === Gemini Families ===
    "gemini-1.5": 100000,
    "gemini-1.5-pro": 100000,
    "gemini-1.5-flash": 100000,
    "gemini-1.5-flash-8b": 100000,
    "gemini-1.5-flash-8b-001": 100000,
    "gemini-1.5-flash-8b-latest": 100000,

    "gemini-2.0": 50000,
    "gemini-2.0-flash": 50000,
    "gemini-2.0-flash-001": 50000,
    "gemini-2.0-flash-exp": 50000,
    "gemini-2.0-flash-lite": 50000,
    "gemini-2.0-flash-lite-001": 50000,
    "gemini-2.0-flash-lite-preview": 50000,
    "gemini-2.0-flash-lite-preview-02-05": 50000,
    "gemini-2.0-flash-thinking-exp": 50000,
    "gemini-2.0-flash-thinking-exp-01-21": 50000,
    "gemini-2.0-flash-thinking-exp-1219": 50000,

    "gemini-2.5": 100000,
    "gemini-2.5-pro": 100000,
    "gemini-2.5-flash": 100000,
    "gemini-2.5-flash-preview-05-20": 100000,
    "gemini-2.5-flash-lite": 100000,
    "gemini-2.5-flash-lite-preview-06-17": 100000,

    # === LearnLM ===
    "learnlm-2.0-flash-experimental": 50000,

    # === Gemma === (typically smaller context, tuned conservatively)
    "gemma-3-1b-it": 4000,
    "gemma-3-4b-it": 4000,
    "gemma-3-12b-it": 4000,
    "gemma-3-27b-it": 4000,
    "gemma-3n-e4b-it": 4000,
    "gemma-3n-e2b-it": 4000,

    # === GPT ===
    "gpt-4": 6000,
    "gpt-4-32k": 24000,
    "gpt-4o": 100000,

    "gpt-3.5": 3000,
    "gpt-3.5-turbo": 3000,
    "gpt-3.5-16k": 12000,
    "gpt-3.5-turbo-16k": 12000,
    "gpt-3.5-32k": 24000,
    "gpt-3.5-turbo-32k": 24000,

    # === Fallback ===
    "default": 32000
}


# Streamlit CSS styles for the app
STREAMLIT_CSS_STYLES = """
    <style>
        /* Main app container */
        .stApp {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        /* Main content area */
        .main .block-container {
            max-width: 100% !important;
            padding: 1rem 1rem 1rem 1rem !important;
        }

        /* Content wrapper */
        .main > div {
            max-width: 100% !important;
        }

        /* Text and code blocks */
        .stMarkdown, .stText, .stCodeBlock, .stDataFrame {
            max-width: 100% !important;
        }

        /* Code and text areas */
        .stCodeBlock pre, .stTextArea textarea, pre, code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            max-width: 100% !important;
        }

        /* Tables */
        .stDataFrame {
            display: block;
            overflow-x: auto;
            width: 100% !important;
        }

        /* Fix for Streamlit's dynamic content */
        [data-testid="stAppViewContainer"] > .main > div {
            padding: 0 !important;
        }
    </style>
"""

