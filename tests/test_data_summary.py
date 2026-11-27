import unittest
import pandas as pd

from src.data_summary import summarize_dataset


class TestDataSummary(unittest.TestCase):

    def test_basic_summary(self):
        df = pd.DataFrame({
            "text": ["Great!", "Terrible!", "Great!"],
            "platform": ["reddit", "reddit", "tiktok"],
        })

        result = summarize_dataset(df)

        self.assertEqual(result["total_posts"], 3)
        self.assertEqual(result["unique_posts"], 2)
        self.assertEqual(result["platform_counts"]["reddit"], 2)
        self.assertEqual(result["platform_counts"]["tiktok"], 1)

    def test_missing_text_count(self):
        df = pd.DataFrame({
            "text": ["Great!", None, "Terrible!"],
        })

        result = summarize_dataset(df)

        self.assertEqual(result["missing_text"], 1)

    def test_sentiment_distribution(self):
        df = pd.DataFrame({
            "text": ["Great!", "Terrible!", "Okay"],
            "label": ["positive", "negative", "neutral"],
        })

        result = summarize_dataset(df)

        self.assertEqual(result["sentiment_counts"]["positive"], 1)
        self.assertEqual(result["sentiment_counts"]["negative"], 1)
        self.assertEqual(result["sentiment_counts"]["neutral"], 1)

    def test_missing_text_column(self):
        df = pd.DataFrame({
            "message": ["Great!"],
        })

        with self.assertRaisesRegex(ValueError, "text"):
            summarize_dataset(df)


if __name__ == "__main__":
    unittest.main()