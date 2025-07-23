import argparse

import pandas as pd

from model.utils.dataset_loader import DatasetLoader


class DataPromptArger:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Process data with AI prompt")
        self.parser.add_argument("--prompt", required=True, help="AI prompt to execute")
        self.parser.add_argument("--dataset", help="Path to dataset (CSV/Parquet). Default Directory: User's Documents")
        self.args = self.parser.parse_args()
        self.df = None

    def get_prompt(self) -> str:
        return self.args.prompt

    def get_dataset(self) -> pd.DataFrame:
        if self.df is None:
            loader = DatasetLoader()
            self.df = loader.load(self.args.dataset)
        return self.df

    def print_df_head(self):
        print(self.get_dataset().head())

    def print_df_shape(self):
        print(f"Dataset shape: {self.get_dataset().shape}")