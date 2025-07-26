import argparse
import pandas as pd

from model.utils.constants import APP_NAME
from model.utils.dataset_loader import DatasetLoader


class DataPromptArger:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=f"🔄 {APP_NAME} CLI: One prompt, one dataset, no chunks."
        )
        subparsers = self.parser.add_subparsers(dest="command", required=True)

        # === Subcommand: new ===
        new_parser = subparsers.add_parser("new", help="Start a new chunking session")
        new_parser.add_argument("--prompt", required=True, help="AI prompt to apply to chunks")
        new_parser.add_argument("--dataset", help="Path to dataset (CSV/Parquet). Default: Documents folder")
        new_parser.add_argument("--chunk-size", type=int, help="Number of rows per chunk (default: 100)")

        # === Subcommand: resume ===
        resume_parser = subparsers.add_parser("resume", help="Resume from previous chunked dataset")

        self.args = self.parser.parse_args()
        self.df = None

    def get_command(self) -> str:
        return self.args.command

    def get_prompt(self) -> str:
        return getattr(self.args, "prompt", "")

    def get_dataset(self) -> pd.DataFrame:
        if self.df is None and hasattr(self.args, "dataset"):
            loader = DatasetLoader()
            self.df = loader.load(self.args.dataset)
        return self.df

    def get_chunk_size(self, default: int = 100) -> int:
        if hasattr(self.args, "chunk_size") and self.args.chunk_size:
            return self.args.chunk_size
        return default

    def print_df_head(self):
        print(self.get_dataset().head())

    def print_df_shape(self):
        print(f"Dataset shape: {self.get_dataset().shape}")
