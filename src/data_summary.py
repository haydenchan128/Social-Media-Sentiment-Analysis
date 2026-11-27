import pandas as pd
import json
from pathlib import Path


def save_dataset_summary(summary: dict, output_path: str) -> None:
    """Save dataset statistics to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)
        

def summarize_dataset(df: pd.DataFrame) -> dict:
    """Generate basic statistics about a social media dataset."""
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

    summary = {
        "total_posts": len(df),
        "missing_text": int(df["text"].isna().sum()),
        "unique_posts": int(df["text"].nunique()),
    }

    if "platform" in df.columns:
        summary["platform_counts"] = (
            df["platform"]
            .value_counts(dropna=False)
            .to_dict()
        )

    if "label" in df.columns:
        summary["sentiment_counts"] = (
            df["label"]
            .value_counts(dropna=False)
            .to_dict()
        )

    return summary