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

    def test_empty_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "empty.csv"
            file_path.write_text("")

            with self.assertRaisesRegex(ValueError, "empty"):
                load_platform_data(file_path, "Reddit")

    def test_csv_with_headers_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "empty.csv"
            file_path.write_text("text,platform\n")

            with self.assertRaisesRegex(ValueError, "no data rows"):
                load_platform_data(file_path, "Reddit")

    def test_malformed_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "malformed.csv"
            file_path.write_text('text,platform\n"Unclosed quote,Reddit\n')

            with self.assertRaisesRegex(ValueError, "Malformed CSV"):
                load_platform_data(file_path, "Reddit")

    def test_column_names_are_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "reddit.csv"

            file_path.write_text(" Text , Username \nHello, user1\n")

            result = load_platform_data(file_path, "Reddit")

            self.assertIn("text", result.columns)
            self.assertIn("username", result.columns)
            self.assertEqual(result["platform"].iloc[0], "reddit")

    def test_duplicate_columns_after_normalization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "duplicate.csv"

            file_path.write_text("text, Text \nHello,Hi\n")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_platform_data(file_path, "Reddit")
                

if __name__ == "__main__":
    unittest.main()
