
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.evaluation import evaluate_predictions
from src.preprocessing import clean_text
from src.sentiment import SentimentAnalyzer


def main():
    dataset = load_dataset(
        "cardiffnlp/tweet_eval",
        "sentiment",
        split="test",
    )

    df = dataset.to_pandas()

    # Dataset labels: 0 = negative, 1 = neutral, 2 = positive
    label_mapping = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    y_true = df["label"].map(label_mapping).tolist()
    analyzer = SentimentAnalyzer()
    
    cleaned_text = df["text"].apply(clean_text)
    results = cleaned_text.apply(analyzer.analyze)

    y_pred = results.apply(lambda result: result["label"]).tolist()

    # Evaluate predictions against dataset labels
    metrics = evaluate_predictions(y_true, y_pred)

    report = metrics["classification_report"]

    print("\nModel performance")
    print("-----------------")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    for label in metrics["labels"]:
        scores = report[label]

        print(f"\n{label.capitalize()}")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F1 score:  {scores['f1-score']:.4f}")

    # Save results locally
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = df[["text", "label"]].copy()
    output["true_label"] = y_true
    output["predicted_label"] = y_pred
    output["compound"] = results.apply(
        lambda result: result["compound"]
    )

    output.to_csv(
        output_dir / "tweeteval_predictions.csv",
        index=False,
    )

    # Save confusion matrix
    confusion = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["labels"],
        columns=metrics["labels"],
    )

    confusion.to_csv(
        output_dir / "confusion_matrix.csv"
    )

    # Save report for later analysis
    with open(
        output_dir / "evaluation_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "dataset": "TweetEval sentiment test split",
                "sample_count": len(df),
                "accuracy": metrics["accuracy"],
                "classification_report": report,
            },
            file,
            indent=2,
        )

    print("\nConfusion matrix:")
    print(confusion)

    print("\nSaved evaluation files to data/processed/")


if __name__ == "__main__":
    main()