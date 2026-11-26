import pandas as pd
from pathlib import Path

from src.preprocessing import clean_text
from src.sentiment import SentimentAnalyzer
from src.validation import validate_input_data
from src.ingestion import load_platform_data, combine_platform_data
from src.logger import setup_logger


logger = setup_logger()


def analyze_dataframe(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Analyze sentiment for a DataFrame and save the results."""

    logger.info("Starting sentiment analysis.")

    try:
        df = validate_input_data(df)

        logger.info("Validated %s posts.", len(df))

        df["clean_text"] = df["text"].apply(clean_text)

        analyzer = SentimentAnalyzer()
        results = df["clean_text"].apply(analyzer.analyze)

        sentiment_df = pd.DataFrame(results.tolist())
        df = pd.concat([df, sentiment_df], axis=1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        logger.info("Processed %s posts.", len(df))
        logger.info("Results saved to %s", output_path)

        return df

    except Exception:
        logger.exception("Sentiment analysis failed.")
        raise


def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """Load one CSV file, analyze sentiment, and save results."""
    df = pd.read_csv(input_path)
    return analyze_dataframe(df, output_path)


def run_multi_platform_pipeline(
    input_files: dict[str, str],
    output_path: str
) -> pd.DataFrame:
    """Load, combine, and analyze data from multiple platforms."""
    if not input_files:
        raise ValueError("No input files provided.")

    datasets = [
        load_platform_data(file_path, platform)
        for platform, file_path in input_files.items()
    ]

    combined_df = combine_platform_data(datasets)

    return analyze_dataframe(combined_df, output_path)


if __name__ == "__main__":
    input_files = {
        "reddit": "data/raw/reddit_sample.csv",
        "tiktok": "data/raw/tiktok_sample.csv",
        "facebook": "data/raw/facebook_sample.csv",
    }

    output_path = "data/processed/multi_platform_results.csv"

    df = run_multi_platform_pipeline(input_files, output_path)

    print(f"Processed {len(df)} posts.")
    print("\nSentiment distribution:")
    print(df["label"].value_counts())
    print(f"\nResults saved to {output_path}")
