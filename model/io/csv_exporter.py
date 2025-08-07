import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from model.utils.constants import CHUNK_JSON_PATH
from model.savers.sqlite_saver import GeminiSQLiteResultSaver


class CSVExporter:
    def __init__(self, json_path: str = CHUNK_JSON_PATH, db_saver: GeminiSQLiteResultSaver = None):
        self.json_path = Path(json_path)
        self.db_saver = db_saver or GeminiSQLiteResultSaver()

    def export_processed_with_original_rows(self, csv_path: str):
        """
        Merge processed rows from SQLite with original data from the JSON file and export to CSV.
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"Chunk JSON file not found at: {self.json_path}")

        # Load chunk data from JSON
        with open(self.json_path, "r") as f:
            data = pd.read_json(f)

        all_chunk_rows = []
        for chunk in data["chunks"]:
            df = pd.DataFrame(chunk["data"])
            df["chunk_id"] = chunk["chunk_id"]
            all_chunk_rows.append(df)

        original_df = pd.concat(all_chunk_rows, ignore_index=True)

        # Load processed rows from DB
        processed_rows: List[Dict[str, Any]] = self.db_saver.get_all()
        processed_df = pd.DataFrame(processed_rows)

        # Merge on source_id
        merged_df = pd.merge(original_df, processed_df, on="source_id", how="inner")

        # Export merged file
        merged_df.to_csv(csv_path, index=False)
        print(f"✅ Exported merged processed CSV to: {csv_path}")
