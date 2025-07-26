import os
import shelve

from model.utils.constants import MODEL_PREFS_DB_PATH, MODEL_KEY


class ModelPreference:
    def __init__(self, db_path=MODEL_PREFS_DB_PATH):
        self.db_path = db_path
        self.key = MODEL_KEY

        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)

    def save_model_name(self, model_name: str):
        with shelve.open(self.db_path) as db:
            db[self.key] = model_name

    def get_model_name(self) -> str:
        with shelve.open(self.db_path) as db:
            return db.get(self.key, '')  # Return empty string if not set

    def clear_model_name(self):
        with shelve.open(self.db_path) as db:
            if self.key in db:
                del db[self.key]
