import os
import shelve
from model.utils.constants import MODEL_PREFS_DB_PATH, MODEL_KEY, MODEL_LIST_KEY, MODEL_CONFIG_KEY, USED_TOKENS_KEY


class ModelPreference:
    def __init__(self, db_path=MODEL_PREFS_DB_PATH):
        self.db_path = db_path
        self.key = MODEL_KEY
        self.list_key = MODEL_LIST_KEY
        self.config_key = MODEL_CONFIG_KEY
        self.token_count_key = USED_TOKENS_KEY

        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)

    # === Token count methods ===
    def save_total_tokens_used(self, token_count: int):
        """Save the total number of tokens used.
        
        Args:
            token_count (int): The total number of tokens used
        """
        with shelve.open(self.db_path) as db:
            db[self.token_count_key] = token_count

    def get_total_tokens_used(self) -> int:
        """Get the total number of tokens used.
        
        Returns:
            int: The total number of tokens used, 0 if not set
        """
        with shelve.open(self.db_path) as db:
            return db.get(self.token_count_key, 0)

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

    # === Per-model generation config ===
    def get_generation_config(self) -> dict:
        """Return saved generation settings for the model or defaults."""
        with shelve.open(self.db_path) as db:
            return db.get(self.config_key, {
                "temperature": 0.2,
                "top_k": 40,
                "top_p": 1.0
            })

    def save_generation_config(self, config: dict):
        """Save generation settings (temperature, top_k, top_p) for a model."""
        with shelve.open(self.db_path) as db:
            db[self.config_key] = config
