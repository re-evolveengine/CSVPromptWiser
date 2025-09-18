import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from utils.constants import RESULTS_DB_PATH


class SQLiteResultSaver:
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
                    timestamp TEXT NOT NULL,
                    UNIQUE(source_id, prompt)  -- Prevent duplicates
                );
            """)
            conn.commit()

    def has_results(self) -> bool:
        """
        Efficiently checks if the 'results' table has any rows.
        Returns True if at least one row exists, False otherwise.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Executes a fast query to see if at least one record exists.
                cursor.execute("SELECT 1 FROM results LIMIT 1;")
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            print(f"Database error while checking for results: {e}")
            return False

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
                    INSERT OR IGNORE INTO results (
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
            print(f"Tried saving {len(results)} rows (duplicates ignored).")

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

    def has_source_ids(self, source_ids: List[str], prompt: str) -> List[str]:
        """
        Return source_ids that already exist in the DB for a given prompt.
        """
        if not source_ids:
            return []

        placeholders = ",".join("?" for _ in source_ids)
        query = f"""
            SELECT source_id FROM results
            WHERE source_id IN ({placeholders}) AND prompt = ?
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (*source_ids, prompt))
            existing = cursor.fetchall()

        return [row[0] for row in existing]


    def clear(self):
        """
        Deletes all records from the 'results' table, resetting it.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM results;")
                conn.commit()
                print("✅ Database results cleared successfully.")
        except sqlite3.Error as e:
            print(f"❌ Database error while clearing results: {e}")
