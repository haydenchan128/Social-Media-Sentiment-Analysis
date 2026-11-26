import pandas as pd
from pathlib import Path


def load_platform_data(file_path: str, platform: str) -> pd.DataFrame:
    """Load and standardize social media data from a CSV file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        df = pd.read_csv(file_path, on_bad_lines="error")

    except pd.errors.EmptyDataError as exc:
        raise ValueError("CSV file is empty or has no header.") from exc

    except pd.errors.ParserError as exc:
        raise ValueError(f"Malformed CSV file: {file_path}") from exc

    except UnicodeDecodeError as exc:
        raise ValueError("CSV file must use a supported text encoding.") from exc

    if df.empty:
        raise ValueError("Dataset contains no data rows.")

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

    df = df.copy()
    df["platform"] = platform.lower().strip()

    return df


def combine_platform_data(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine datasets from multiple social media platforms."""

    if not datasets:
        raise ValueError("No datasets provided.")

    return pd.concat(datasets, ignore_index=True)