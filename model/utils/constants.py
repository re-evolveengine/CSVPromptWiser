import os

JSON_CHUNK_VERSION = 1.0
APP_NAME = "Chunkwise"
DOCUMENTS_PATH = os.path.join(os.path.expanduser("~"), "Documents")
APP_PATH = os.path.join(DOCUMENTS_PATH, APP_NAME)
DEFAULT_CHUNK_SIZE = 100