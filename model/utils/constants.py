import os

JSON_CHUNK_VERSION = 1.0
APP_NAME = "Chunkwise"
DATA_FOLDER_NAME = "data"
RESULTS_FOLDER_NAME = "results"
TEMP_FOLDER_NAME = "temp"
DOCUMENTS_DIR = os.path.join(os.path.expanduser("~"), "Documents")
APP_DIR = os.path.join(DOCUMENTS_DIR, APP_NAME)
DATA_DIR = os.path.join(APP_DIR, DATA_FOLDER_NAME)
RESULTS_DIR = os.path.join(APP_DIR, RESULTS_FOLDER_NAME)
TEMP_DIR = os.path.join(APP_DIR, TEMP_FOLDER_NAME)
DEFAULT_CHUNK_SIZE = 100