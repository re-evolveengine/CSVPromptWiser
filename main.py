import argparse

from model.utils.data_prompt_arger import DataPromptArger
from model.utils.dataset_loader import DatasetLoader
import pandas as pd


def main():
    arger = DataPromptArger()
    
    try:
        print(f"\nPrompt: {arger.get_prompt()}")
        arger.print_df_head()
        arger.print_df_shape()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()