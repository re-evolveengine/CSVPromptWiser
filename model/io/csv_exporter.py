# csv_exporter.py

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from model.io.sqlite_result_saver import SQLiteResultSaver
from utils.constants import JSON_CHUNK_FILE


class CSVExporter:
    def __init__(self, json_path: str = JSON_CHUNK_FILE, db_saver: SQLiteResultSaver = None):
        self.json_path = Path(json_path)
        self.db_saver = db_saver or SQLiteResultSaver()

    def export_processed_with_original_rows(self, csv_path: str):
        """
        Merge processed rows from SQLite with original data from the JSON file and export to CSV.
        """
        processed_rows: List[Dict[str, Any]] = self.db_saver.get_all()
        if not processed_rows:
            raise ValueError("No processed data found in the database to export.")
        processed_df = pd.DataFrame(processed_rows)

        if not self.json_path.exists():
            print(f"Warning: Chunk JSON file not found at {self.json_path}. Exporting only data from the database.")
            merged_df = processed_df
        else:
            with open(self.json_path, "r") as f:
                data = json.load(f)

            all_chunk_rows = []
            for chunk in data["chunks"]:
                df = pd.DataFrame(chunk["data"])
                df["chunk_id"] = chunk["chunk_id"]
                all_chunk_rows.append(df)

            original_df = pd.concat(all_chunk_rows, ignore_index=True)
            merged_df = pd.merge(original_df, processed_df, on="source_id", how="right")

        # --- NEW CODE TO CLEAN UP DUPLICATE COLUMNS ---
        # Check if the duplicate columns exist after the merge
        if 'chunk_id_x' in merged_df.columns and 'chunk_id_y' in merged_df.columns:
            # Drop the column from the original file ('_x')
            merged_df = merged_df.drop(columns=['chunk_id_x'])
            # Rename the column from the database ('_y') to the clean name
            merged_df = merged_df.rename(columns={'chunk_id_y': 'chunk_id'})

        # Export the final, cleaned-up file
        merged_df.to_csv(csv_path, index=False)
        print(f"✅ Exported {len(merged_df)} rows to: {csv_path}")