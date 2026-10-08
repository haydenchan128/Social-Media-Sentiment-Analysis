import pandas as pd

from preprocessing import clean_text
from sentiment import SentimentAnalyzer


def run_pipeline(input_path: str, output_path: str):
    """
    Run the complete sentiment analysis pipeline.
    """

    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

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
    run_pipeline(
        "data/raw/social_media.csv",
        "data/processed/sentiment_results.csv"
    )