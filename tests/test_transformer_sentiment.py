import unittest

from src.transformer_sentiment import (
    TransformerSentimentAnalyzer,
    preprocess_for_transformer,
)


class TestTransformerSentiment(unittest.TestCase):

    def test_preprocess_for_transformer(self):
        text = "@someone Check this https://example.com"

        result = preprocess_for_transformer(text)

        self.assertEqual(result, "@user Check this http")

    def test_analyze_many(self):
        class MockClassifier:
            def __call__(self, texts, batch_size, truncation):
                return [
                    {"label": "positive", "score": 0.95},
                    {"label": "negative", "score": 0.88},
                ]

        analyzer = TransformerSentimentAnalyzer(
            classifier=MockClassifier()
        )

        results = analyzer.analyze_many([
            "This is great!",
            "This is terrible!",
        ])

        self.assertEqual(len(results), 2)

        self.assertEqual(results[0]["label"], "positive")
        self.assertEqual(results[0]["confidence"], 0.95)

        self.assertEqual(results[1]["label"], "negative")
        self.assertEqual(results[1]["confidence"], 0.88)

    def test_analyze_many_empty_input(self):
        class MockClassifier:
            def __call__(self, texts, batch_size, truncation):
                return []

        analyzer = TransformerSentimentAnalyzer(
            classifier=MockClassifier()
        )

        results = analyzer.analyze_many([])

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()