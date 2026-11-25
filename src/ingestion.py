
import pandas as pd
from pathlib import Path


def load_platform_data(
    file_path: str,
    platform: str,
) -> pd.DataFrame:
    """Load and standardize social media data from a CSV file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

    df = df.copy()

    df["platform"] = platform.lower().strip()

    return df


def combine_platform_data(
    datasets: list[pd.DataFrame],
) -> pd.DataFrame:
    """Combine datasets from multiple social media platforms."""

    if not datasets:
        raise ValueError("No datasets provided.")

    combined_df = pd.concat(
        datasets,
        ignore_index=True,
    )

    return combined_df
