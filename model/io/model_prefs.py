import os
import shelve

from model.utils.constants import MODEL_PREFS_DB_PATH, MODEL_KEY, MODEL_LIST_KEY


class ModelPreference:
    def __init__(self, db_path=MODEL_PREFS_DB_PATH):
        self.db_path = db_path
        self.key = MODEL_KEY
        self.list_key = MODEL_LIST_KEY  # New key for list

        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)

    # === Single model (selected) ===
    def save_selected_model_name(self, model_name: str):
        with shelve.open(self.db_path) as db:
            db[self.key] = model_name

    def get_selected_model_name(self) -> str:
        with shelve.open(self.db_path) as db:
            return db.get(self.key, '')

    def clear_selected_model_name(self):
        with shelve.open(self.db_path) as db:
            if self.key in db:
                del db[self.key]

    # === List of models ===
    def save_model_list(self, model_list: list[str]):
        with shelve.open(self.db_path) as db:
            db[self.list_key] = model_list

    def get_model_list(self) -> list[str]:
        with shelve.open(self.db_path) as db:
            return db.get(self.list_key, [])

    def clear_model_list(self):
        with shelve.open(self.db_path) as db:
            if self.list_key in db:
                del db[self.list_key]
