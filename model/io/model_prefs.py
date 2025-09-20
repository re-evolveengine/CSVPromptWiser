import os
import shelve
from contextlib import contextmanager
from typing import List, Dict

from utils.constants import MODEL_PREFS_DB_PATH, MODEL_KEY, MODEL_LIST_KEY, MODEL_CONFIG_KEY, \
    REMAINING_TOTAL_TOKENS_KEY, TOTAL_TOKENS_KEY, CHUNK_SIZE_KEY, DEFAULT_CHUNK_SIZE


class ModelPreference:
    def __init__(self, db_path: str = MODEL_PREFS_DB_PATH):
        """Initialize ModelPreference with database path and keys.
        
        Args:
            db_path: Path to the shelve database file
        """
        self.db_path = db_path
        self.key = MODEL_KEY
        self.list_key = MODEL_LIST_KEY
        self.config_key = MODEL_CONFIG_KEY
        self.remaining_tokens_key = REMAINING_TOTAL_TOKENS_KEY
        self.total_tokens_key = TOTAL_TOKENS_KEY
        self.chunk_size_key = CHUNK_SIZE_KEY
        self._ensure_db_dir()

    def _ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)

    @contextmanager
    def _shelve_operation(self, mode: str = 'c') -> shelve.Shelf:
        """Context manager for shelve database operations.
        
        Args:
            mode: The mode to open the shelve database
            
        Yields:
            The opened shelve database
        """
        db = shelve.open(self.db_path, mode)
        try:
            yield db
        finally:
            db.close()


    @property
    def chunk_size(self) -> int:
        """int: Get or set the chunk size."""
        with self._shelve_operation() as db:
            return db.get(self.chunk_size_key, DEFAULT_CHUNK_SIZE)

    @chunk_size.setter
    def chunk_size(self, chunk_size: int) -> None:
        with self._shelve_operation() as db:
            db[self.chunk_size_key] = chunk_size

    # === Token count properties ===
    @property
    def remaining_total_tokens(self) -> int:
        """int: Get or set the remaining total tokens."""
        with self._shelve_operation() as db:
            return db.get(self.remaining_tokens_key, 0)

    @remaining_total_tokens.setter
    def remaining_total_tokens(self, token_count: int) -> None:
        with self._shelve_operation() as db:
            db[self.remaining_tokens_key] = token_count

    @property
    def total_tokens(self) -> int:
        """int: Get or set the total tokens used."""
        with self._shelve_operation() as db:
            return db.get(self.total_tokens_key, 0)

    @total_tokens.setter
    def total_tokens(self, token_count: int) -> None:
        with self._shelve_operation() as db:
            db[self.total_tokens_key] = token_count

    # === Model selection properties ===
    @property
    def selected_model_name(self) -> str:
        """str: Get or set the currently selected model name."""
        with self._shelve_operation() as db:
            return db.get(self.key, '')

    @selected_model_name.setter
    def selected_model_name(self, model_name: str) -> None:
        with self._shelve_operation() as db:
            db[self.key] = model_name

    @selected_model_name.deleter
    def selected_model_name(self) -> None:
        """Remove the selected model name from the database."""
        with self._shelve_operation() as db:
            if self.key in db:
                del db[self.key]

    # === Model list properties ===
    @property
    def model_list(self) -> List[str]:
        """List[str]: Get or set the list of available models."""
        with self._shelve_operation() as db:
            return db.get(self.list_key, [])

    @model_list.setter
    def model_list(self, model_list: List[str]) -> None:
        with self._shelve_operation() as db:
            db[self.list_key] = model_list

    @model_list.deleter
    def model_list(self) -> None:
        """Remove the model list from the database."""
        with self._shelve_operation() as db:
            if self.list_key in db:
                del db[self.list_key]

    # === Generation config properties ===
    @property
    def generation_config(self) -> Dict[str, float]:
        """Dict[str, float]: Get or set the generation configuration.
        
        Returns:
            Default configuration if none is set:
            {
                "temperature": 0.2,
                "top_k": 40,
                "top_p": 1.0
            }
        """
        with self._shelve_operation() as db:
            return db.get(self.config_key, {
                "temperature": 0.2,
                "top_k": 40,
                "top_p": 1.0
            })

    @generation_config.setter
    def generation_config(self, config: Dict[str, float]) -> None:
        with self._shelve_operation() as db:
            db[self.config_key] = config
