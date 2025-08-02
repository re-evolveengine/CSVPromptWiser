import os

JSON_CHUNK_VERSION = 1.0
APP_NAME = "ChunkWisePrompter"
DATA_FOLDER_NAME = "data"
RESULTS_FOLDER_NAME = "results"
TEMP_FOLDER_NAME = "temp"
CONFIG_FOLDER_NAME = "config"  # Optional: for preferences/configs

DOCUMENTS_DIR = os.path.join(os.path.expanduser("~"), "Documents")
APP_DIR = os.path.join(DOCUMENTS_DIR, APP_NAME)
DATA_DIR = os.path.join(APP_DIR, DATA_FOLDER_NAME)
RESULTS_DIR = os.path.join(APP_DIR, RESULTS_FOLDER_NAME)
TEMP_DIR = os.path.join(APP_DIR, TEMP_FOLDER_NAME)
CONFIG_DIR = os.path.join(APP_DIR, CONFIG_FOLDER_NAME)

DEFAULT_CHUNK_SIZE = 100

# 🗄️ Model preference DB (shelve) file name and full path
MODEL_PREFS_DB_NAME = "model_prefs.db"
MODEL_PREFS_DB_PATH = os.path.join(CONFIG_DIR, MODEL_PREFS_DB_NAME)
MODEL_KEY = "gemini_model"
MODEL_LIST_KEY = "gemini_model_list"
MODEL_CONFIG_KEY = "gemini_model_config"

# Generation behavior
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 1.0


# CLI constants
APP_NAME_CLI = "PromptPilot"
APP_DIR_CLI = os.path.join(DOCUMENTS_DIR, APP_NAME_CLI)
DATA_DIR_CLI = os.path.join(APP_DIR_CLI, DATA_FOLDER_NAME)
RESULTS_DIR_CLI = os.path.join(APP_DIR_CLI, RESULTS_FOLDER_NAME)
TEMP_DIR_CLI = os.path.join(APP_DIR_CLI, TEMP_FOLDER_NAME)
CONFIG_DIR_CLI = os.path.join(APP_DIR_CLI, CONFIG_FOLDER_NAME)
MODEL_PREFS_DB_PATH_CLI = os.path.join(CONFIG_DIR_CLI, MODEL_PREFS_DB_NAME)

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
    "gemini-1.5-flash-8b-001": 1048576,
    "gemini-1.5-flash-8b-latest": 1048576,
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
