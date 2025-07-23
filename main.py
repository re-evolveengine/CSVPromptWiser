# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from dotenv import load_dotenv
import os
import argparse

def print_geni(name):

    load_dotenv()  # Loads from .env
    gemini_key = os.getenv("GEMINI_API_KEY")
    print(gemini_key)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='world', help='Name to greet')
    return parser.parse_args()

import argparse
import pandas as pd

def get_prompt_dataset():
    # Set up CLI parser
    parser = argparse.ArgumentParser(description="Process a dataset with an AI prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="AI prompt to apply")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (CSV/Excel)")
    args = parser.parse_args()

    # Load data (supports CSV/Excel)
    df = pd.read_csv(args.dataset) if args.dataset.endswith('.csv') else pd.read_excel(args.dataset)

    # Output
    print(f"\nPrompt: '{args.prompt}'\n")
    print("Dataset Head:")
    print(df.head())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_prompt_dataset()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
