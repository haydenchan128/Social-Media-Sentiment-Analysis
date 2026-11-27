import pandas as pd
from pathlib import Path
import argparse

from src.preprocessing import clean_text
from src.sentiment import SentimentAnalyzer
from src.validation import validate_input_data
from src.ingestion import load_platform_data, combine_platform_data
from src.logger import setup_logger
from src.deduplication import remove_duplicate_posts
from src.data_summary import summarize_dataset, save_dataset_summary



logger = setup_logger()


def analyze_dataframe(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Analyze sentiment for a DataFrame and save the results."""

    logger.info("Starting sentiment analysis.")

    try:
        df = validate_input_data(df)
        logger.info("Validated %s posts.", len(df))

        original_count = len(df)

        df = remove_duplicate_posts(df)

        duplicates_removed = original_count - len(df)
        remaining_count = len(df)

        logger.info("Original post count: %s", original_count)
        logger.info("Duplicate posts removed: %s", duplicates_removed)
        logger.info("Posts remaining: %s", remaining_count)

        summary = summarize_dataset(df)
        logger.info("Dataset summary: %s", summary)

        summary_path = Path("data/processed/dataset_summary.json")
        save_dataset_summary(summary, summary_path)
        logger.info("Dataset summary saved to %s", summary_path)    

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

    try:
        logger.info("Loading dataset from %s", input_path)
        df = pd.read_csv(input_path, on_bad_lines="error")
        logger.info("Successfully loaded dataset.")
        return analyze_dataframe(df, output_path)

    except pd.errors.EmptyDataError as exc:
        logger.exception("CSV file is empty: %s", input_path)
        raise ValueError("CSV file is empty or has no header.") from exc

    except pd.errors.ParserError as exc:
        logger.exception("Malformed CSV file: %s", input_path)
        raise ValueError(f"Malformed CSV file: {input_path}") from exc

    except UnicodeDecodeError as exc:
        logger.exception("Unable to decode CSV file: %s", input_path)
        raise ValueError("CSV file has an unsupported text encoding.") from exc

    except Exception:
        logger.exception("Failed to run pipeline on %s", input_path)
        raise


def run_multi_platform_pipeline(
    input_files: dict[str, str],
    output_path: str
) -> pd.DataFrame:
    """Load, combine, and analyze data from multiple platforms."""

    if not input_files:
        raise ValueError("No input files provided.")

    try:
        logger.info(
            "Starting multi-platform pipeline with %s datasets.",
            len(input_files)
        )

        datasets = [
            load_platform_data(file_path, platform)
            for platform, file_path in input_files.items()
        ]

        logger.info("Successfully loaded all platform datasets.")
        combined_df = combine_platform_data(datasets)
        logger.info("Combined %s posts.", len(combined_df))
        return analyze_dataframe(combined_df, output_path)

    except Exception:
        logger.exception("Multi-platform pipeline failed.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run social media sentiment analysis."
    )

    parser.add_argument(
        "--reddit",
        help="Path to the Reddit CSV file."
    )

    parser.add_argument(
        "--tiktok",
        help="Path to the TikTok CSV file."
    )

    parser.add_argument(
        "--facebook",
        help="Path to the Facebook CSV file."
    )

    parser.add_argument(
        "--output",
        default="data/processed/multi_platform_results.csv",
        help="Path to save the sentiment analysis results."
    )

    args = parser.parse_args()

    input_files = {
        platform: file_path
        for platform, file_path in {
            "reddit": args.reddit,
            "tiktok": args.tiktok,
            "facebook": args.facebook,
        }.items()
        if file_path is not None
    }

    if not input_files:
        parser.error("Provide at least one platform CSV file.")

    if len(input_files) == 1:
        platform, input_path = next(iter(input_files.items()))
        logger.info("Running single-platform pipeline: %s", platform)
        df = run_pipeline(input_path, args.output)
    else:
        logger.info("Running multi-platform pipeline.")
        df = run_multi_platform_pipeline(input_files, args.output)

    print(f"Processed {len(df)} posts.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
