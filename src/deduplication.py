import pandas as pd


def remove_duplicate_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate posts while preserving the first occurrence."""
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

    normalized_text = df["text"].map(
        lambda text: " ".join(text.split()).casefold()
    )

    if "platform" in df.columns:
        platform = (
            df["platform"]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        duplicate_mask = pd.DataFrame({
            "platform": platform,
            "text": normalized_text,
        }).duplicated()
    else:
        duplicate_mask = normalized_text.duplicated()

    return df.loc[~duplicate_mask].reset_index(drop=True).copy()