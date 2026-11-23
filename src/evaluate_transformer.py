
import json
import os

import pandas as pd
from datasets import load_dataset
from transformers import pipeline

from src.evaluation import evaluate_predictions


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = "data/processed"


def preprocess_for_transformer(text: str) -> str:
    """Prepare social media text using the model's expected conventions."""
    import re

    text = str(text).strip()
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    return text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading TweetEval test dataset...")
    dataset = load_dataset(
        "cardiffnlp/tweet_eval",
        "sentiment",
        split="test",
    )

    label_mapping = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    y_true = [label_mapping[label] for label in dataset["label"]]
    texts = [
        preprocess_for_transformer(text)
        for text in dataset["text"]
    ]

    print("Loading transformer model...")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )

    print("Predicting sentiment...")
    predictions = classifier(
        texts,
        batch_size=16,
        truncation=True,
    )

    y_pred = [item["label"].lower() for item in predictions]
    confidence = [item["score"] for item in predictions]

    print("Calculating evaluation metrics...")
    metrics = evaluate_predictions(y_true, y_pred)

    # Save individual predictions.
    results_df = pd.DataFrame({
        "text": dataset["text"],
        "true_label": y_true,
        "predicted_label": y_pred,
        "confidence": confidence,
    })
    results_df.to_csv(
        f"{OUTPUT_DIR}/transformer_predictions.csv",
        index=False,
    )

    # Save metrics and the classification report.
    metrics_to_save = {
        "model": MODEL_NAME,
        "accuracy": float(metrics["accuracy"]),
        "classification_report": metrics["classification_report"],
        "labels": metrics["labels"],
    }
    with open(
        f"{OUTPUT_DIR}/transformer_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics_to_save, file, indent=4)

    # Save the confusion matrix with named rows and columns.
    matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["labels"],
        columns=metrics["labels"],
    )
    matrix_df.index.name = "actual"
    matrix_df.to_csv(
        f"{OUTPUT_DIR}/transformer_confusion_matrix.csv"
    )

    # Print a summary.
    print("\nTransformer Model Performance")
    print("----------------------------")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    for label in metrics["labels"]:
        scores = metrics["classification_report"][label]
        print(
            f"{label.capitalize():<10}"
            f"Precision: {scores['precision']:.4f}  "
            f"Recall: {scores['recall']:.4f}  "
            f"F1: {scores['f1-score']:.4f}"
        )

    print("\nConfusion Matrix:")
    print(matrix_df)

    print(f"\nResults saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()