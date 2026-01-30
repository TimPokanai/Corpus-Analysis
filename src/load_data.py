import pandas as pd
from pathlib import Path

def load_prompts():
    data_path = Path("../data/raw/prompts.csv")

    df = pd.read_csv(data_path, encoding="utf-8")

    return df

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    required_columns = {"prompt", "for_devs"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Removing rows with empty missing or values in the 'prompt' column
    df = df.dropna(subset="prompt")

    # Ensuring 'prompt' column is of type String
    df["prompt"] = df["prompt"].astype(str)

    return df

if __name__ == "__main__":
    prompts_df = load_prompts()

    print(prompts_df.head())
    print(prompts_df.columns)
    print(f"Number of rows: {len(prompts_df)}")
