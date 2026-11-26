import pandas as pd

REQUIRED_COLUMNS = {"text"}


def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean input data before sentiment analysis."""

    if df.empty:
        raise ValueError("Input dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing_columns))}"
        )

    # Reject non-string values
    if not df["text"].dropna().map(
        lambda value: isinstance(value, str)
    ).all():
        raise ValueError("All non-missing values in the text column must be strings.")

    # Remove missing and blank text
    valid_text = (
        df["text"].notna()
        & df["text"].str.strip().ne("")
    )

    df = df.loc[valid_text].copy()

    if df.empty:
        raise ValueError("No usable text values found in the dataset.")

    return df.reset_index(drop=True)