import pandas as pd

from src.preprocessing import clean_text
from src.sentiment import SentimentAnalyzer
from src.validation import validate_input_data


def run_pipeline(input_path: str, output_path: str):
    """Run the complete sentiment analysis pipeline."""

    df = pd.read_csv(input_path)
    df = validate_input_data(df)

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Sentiment analysis
    analyzer = SentimentAnalyzer()
    results = df["clean_text"].apply(analyzer.analyze)
    sentiment_df = pd.DataFrame(results.tolist())

    df = pd.concat([df, sentiment_df], axis=1)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    output_path = "data/processed/sentiment_results.csv"

    import os
    os.makedirs("data/processed", exist_ok=True)

    df = run_pipeline(
        "data/raw/social_media.csv",
        output_path,
    )

    print(f"Processed {len(df)} posts.")
    print("\nSentiment distribution:")
    print(df["label"].value_counts())
    print(f"\nResults saved to {output_path}")