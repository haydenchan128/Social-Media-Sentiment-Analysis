import pandas as pd


def sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall sentiment distribution.
    """

    summary = (
        df["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    )

    summary.columns = ["sentiment", "percentage"]

    return summary


def platform_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sentiment distribution by platform.
    """

    return pd.crosstab(
        df["platform"],
        df["label"],
        normalize="index"
    ).mul(100).round(2)


def brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average sentiment by brand.
    """

    return (
        df.groupby("brand")["compound"]
        .mean()
        .sort_values(ascending=False)
    )