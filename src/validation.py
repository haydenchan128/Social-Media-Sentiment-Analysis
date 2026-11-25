import pandas as pd


REQUIRED_COLUMNS = {"text"}


def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input data before sentiment analysis."""

    if df.empty:
        raise ValueError("Input dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing_columns))}"
        )

    if df["text"].isna().all():
        raise ValueError("The text column contains no usable values.")

    if not df["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("All values in the text column must be strings.")

    if df["text"].str.strip().eq("").all():
        raise ValueError("The text column contains only empty strings.")

    return df.copy()
