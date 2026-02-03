import pandas as pd
from pathlib import Path

from preprocess import preprocess_documents_by_category

RAW_DATA_PATH = Path("../data/raw/bbc-text.csv")

TEXT_COLUMN = "text"
LABEL_COLUMN = "category"

LABELS_MAP = {
    0: "tech",
    1: "business",
    2: "sport",
    3: "entertainment",
    4: "politics"
}

def load_raw_csv():

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data csv file not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")

    return df

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Removing rows with empty missing or values in the 'text' column
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df[df[TEXT_COLUMN].str.strip().astype(bool)]

    # Extract numeric label from LABEL_COLUMN (i.e.: "0 tech" -> 0)
    df[LABEL_COLUMN] = (
    df[LABEL_COLUMN]
    .astype(str)
    .str.lower()
    .str.replace(r"\d+", "", regex=True)        # remove numbers
    .str.replace(r"[^a-z\s]", "", regex=True)   # remove special chars
    .str.strip()                                # trim whitespace
)

    invalid_labels = set(df[LABEL_COLUMN].unique()) - set(LABELS_MAP.values())
    if invalid_labels:
        raise ValueError(f"Unexpected label values found: {invalid_labels}")

    return df

def get_all_documents(df: pd.DataFrame) -> list[str]:
    return df[TEXT_COLUMN].tolist()

def get_documents_by_category(df: pd.DataFrame) -> dict[str, list[str]]:
    grouped_docs = {name: [] for name in LABELS_MAP.values()}

    for _, row in df.iterrows():
        grouped_docs[row[LABEL_COLUMN]].append(row[TEXT_COLUMN])

    return grouped_docs

def compute_dataset_stats(df: pd.DataFrame) -> dict:
    stats = {
        "total_documents": len(df),
        "documents_per_category": {},
    }

    for label_name in LABELS_MAP.values():
        count = (df[LABEL_COLUMN] == label_name).sum()
        stats["documents_per_category"][label_name] = int(count)

    return stats
