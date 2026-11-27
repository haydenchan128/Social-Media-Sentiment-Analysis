import re

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def preprocess_for_transformer(text: str) -> str:
    """Prepare social media text using the model's expected conventions."""
    text = str(text).strip()
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    return text


class TransformerSentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, classifier=None):
        self.model_name = model_name

        if classifier is None:
            from transformers import pipeline

            classifier = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
        )

        self.classifier = classifier

    def analyze_many(self, texts: list[str]) -> list[dict]:
        """Analyze multiple posts using batch predictions."""
        processed_texts = [
            preprocess_for_transformer(text)
            for text in texts
        ]

        predictions = self.classifier(
            processed_texts,
            batch_size=16,
            truncation=True,
        )

        return [
            {
                "label": prediction["label"].lower(),
                "confidence": float(prediction["score"]),
            }
            for prediction in predictions
        ]