import unittest
import pandas as pd
import json
import tempfile
from pathlib import Path

from src.data_summary import summarize_dataset, save_dataset_summary


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

    def test_save_dataset_summary(self):
        summary = {
            "total_posts": 10,
            "missing_text": 0,
            "unique_posts": 8,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.json"

            save_dataset_summary(summary, output_path)

            self.assertTrue(output_path.exists())

            with output_path.open(encoding="utf-8") as file:
                saved_summary = json.load(file)

            self.assertEqual(saved_summary, summary)

    def test_save_summary_creates_parent_directory(self):
        summary = {"total_posts": 5}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "summary.json"

            save_dataset_summary(summary, output_path)

            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()