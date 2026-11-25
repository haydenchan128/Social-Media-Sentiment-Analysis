
import unittest
import pandas as pd
import tempfile

from pathlib import Path

from src.ingestion import (
    load_platform_data,
    combine_platform_data,
)


class TestIngestion(unittest.TestCase):

    def test_load_platform_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "reddit.csv"

            pd.DataFrame({
                "text": ["I love this!"]
            }).to_csv(file_path, index=False)

            result = load_platform_data(
                str(file_path),
                "Reddit",
            )

            self.assertEqual(result.loc[0, "platform"], "reddit")
            self.assertEqual(len(result), 1)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_platform_data("missing.csv", "reddit")

    def test_missing_text_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "invalid.csv"

            pd.DataFrame({
                "message": ["Hello"]
            }).to_csv(file_path, index=False)

            with self.assertRaises(ValueError):
                load_platform_data(str(file_path), "reddit")

    def test_combine_platform_data(self):
        reddit = pd.DataFrame({
            "text": ["Reddit post"],
            "platform": ["reddit"],
        })

        tiktok = pd.DataFrame({
            "text": ["TikTok post"],
            "platform": ["tiktok"],
        })

        result = combine_platform_data([reddit, tiktok])

        self.assertEqual(len(result), 2)
        self.assertEqual(
            set(result["platform"]),
            {"reddit", "tiktok"},
        )

    def test_empty_dataset_list(self):
        with self.assertRaises(ValueError):
            combine_platform_data([])


if __name__ == "__main__":
    unittest.main()
