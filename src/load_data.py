import pandas as pd
from pathlib import Path

def load_prompts():
    data_path = Path("../data/raw/prompts.csv")

    df = pd.read_csv(data_path, encoding="utf-8")

    return df

if __name__ == "__main__":
    prompts_df = load_prompts()

    print(prompts_df.head())
    print(prompts_df.columns)
    print(f"Number of rows: {len(prompts_df)}")
