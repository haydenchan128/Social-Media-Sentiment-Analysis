import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

def sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall sentiment distribution.
    """

    if df.empty:
        return pd.DataFrame(columns=["sentiment", "percentage"])

    summary = (
        df["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename_axis("sentiment")
        .reset_index(name="percentage")
    )

    summary.columns = ["sentiment", "percentage"]

    return summary


def platform_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sentiment distribution by platform.
    """

    if df.empty:
        return pd.DataFrame()

    return pd.crosstab(
        df["platform"],
        df["label"],
        normalize="index"
    ).mul(100).round(2)


def brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average sentiment by brand.
    """

    if df.empty:
        return pd.Series(dtype=float, name="compound")

    return (
        df.groupby("brand")["compound"]
        .mean()
        .sort_values(ascending=False)
    )


def evaluate_predictions(y_true, y_pred) -> dict:
    """Compare predicted sentiment labels with human-labeled labels."""

    labels = ["negative", "neutral", "positive"]

    if len(y_true) == 0:
        raise ValueError("Evaluation requires at least one labeled example.")

    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predictions must have equal lengths.")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ),
        "labels": labels,
    }