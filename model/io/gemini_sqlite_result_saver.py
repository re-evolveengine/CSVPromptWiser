import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from model.utils.constants import RESULTS_DB_PATH


class GeminiSQLiteResultSaver:
    def __init__(self, db_path: str = RESULTS_DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)
            conn.commit()

    def save(self, results: List[Dict[str, Any]]):
        """
        Save a list of results. Each result must include:
        - 'source_id': str
        - 'prompt': str
        - 'response': str
        """
        if not results:
            raise ValueError("No results to save.")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for item in results:
                source_id = item["source_id"]
                prompt = item["prompt"]
                response = item["response"]
                timestamp = datetime.utcnow().isoformat() + "Z"

                cursor.execute("""
                    INSERT INTO results (source_id, prompt, response, timestamp)
                    VALUES (?, ?, ?, ?);
                """, (source_id, prompt, response, timestamp))

            conn.commit()
            print(f"Saved {len(results)} results to {self.db_path}")

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all saved results as a list of dictionaries.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, source_id, prompt, response, timestamp FROM results")
            rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "source_id": row[1],
                "prompt": row[2],
                "response": row[3],
                "timestamp": row[4]
            }
            for row in rows
        ]
