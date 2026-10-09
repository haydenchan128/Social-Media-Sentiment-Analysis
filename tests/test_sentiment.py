import unittest
import pandas as pd
from src.evaluation import (
    sentiment_summary,
    platform_summary,
    brand_summary,
    evaluate_predictions,
)

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "platform": ["Reddit", "Reddit", "TikTok", "TikTok"],
            "brand": ["BrandA", "BrandA", "BrandB", "BrandB"],
            "label": ["positive", "negative", "positive", "neutral"],
            "compound": [0.8, -0.7, 0.6, 0.0],
        })

    def test_sentiment_summary(self):
        result = sentiment_summary(self.df)

        self.assertAlmostEqual(
            result["percentage"].sum(), 100.0
        )

    def test_platform_summary(self):
        result = platform_summary(self.df)

        self.assertIn("Reddit", result.index)
        self.assertIn("TikTok", result.index)
        self.assertAlmostEqual(result.loc["Reddit"].sum(), 100.0)

    def test_brand_summary(self):
        result = brand_summary(self.df)

        self.assertGreater(
            result["BrandB"], result["BrandA"]
        )

    def test_perfect_predictions(self):
        labels = ["positive", "negative", "neutral"]
        result = evaluate_predictions(labels, labels)

        self.assertEqual(result["accuracy"], 1.0)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(
                ["positive", "negative"],
                ["positive"],
            )


if __name__ == "__main__":
    unittest.main()