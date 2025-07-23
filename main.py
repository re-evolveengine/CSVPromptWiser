import argparse
from model.utils.dataset_loader import DatasetLoader
import pandas as pd


class DataPromptArg:
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


def main():
    analyzer = DataPromptArg()
    
    try:
        print(f"\nPrompt: {analyzer.get_prompt()}")
        analyzer.print_df_head()
        analyzer.print_df_shape()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()