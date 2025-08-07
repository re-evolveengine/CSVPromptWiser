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
                    chunk_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    used_tokens INTEGER,
                    model_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)
            conn.commit()

    def save(self, results: List[Dict[str, Any]]):
        """
        Save a list of processed rows to the database.

        Each result dict must include:
        - 'source_id': str
        - 'chunk_id': str
        - 'prompt': str
        - 'response': str
        - 'model_version': str
        - 'used_tokens': int or None
        """
        if not results:
            raise ValueError("No results to save.")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for item in results:
                cursor.execute("""
                    INSERT INTO results (
                        source_id,
                        chunk_id,
                        prompt,
                        response,
                        used_tokens,
                        model_version,
                        timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (
                    item["source_id"],
                    item["chunk_id"],
                    item["prompt"],
                    item["response"],
                    item.get("used_tokens"),
                    item["model_version"],
                    datetime.utcnow().isoformat() + "Z"
                ))

            conn.commit()
            print(f"Saved {len(results)} results to {self.db_path}")

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all saved results.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source_id, chunk_id, prompt, response, used_tokens, model_version, timestamp
                FROM results
            """)
            rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "source_id": row[1],
                "chunk_id": row[2],
                "prompt": row[3],
                "response": row[4],
                "used_tokens": row[5],
                "model_version": row[6],
                "timestamp": row[7],
            }
            for row in rows
        ]
