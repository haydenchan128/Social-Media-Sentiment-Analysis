import unittest
import pandas as pd

from src.deduplication import remove_duplicate_posts


class TestDeduplication(unittest.TestCase):

    def test_removes_duplicate_posts(self):
        df = pd.DataFrame({
            "text": [
                "I love this!",
                "I love this!",
                "This is great!",
            ],
            "platform": ["reddit", "reddit", "reddit"],
        })

        result = remove_duplicate_posts(df)

        self.assertEqual(len(result), 2)

    def test_ignores_case_and_extra_whitespace(self):
        df = pd.DataFrame({
            "text": [
                "This is great!",
                "  THIS IS GREAT!  ",
            ],
            "platform": ["reddit", "reddit"],
        })

        result = remove_duplicate_posts(df)

        self.assertEqual(len(result), 1)

    def test_keeps_identical_posts_from_different_platforms(self):
        df = pd.DataFrame({
            "text": ["Great post!", "Great post!"],
            "platform": ["reddit", "tiktok"],
        })

        result = remove_duplicate_posts(df)

        self.assertEqual(len(result), 2)

    def test_preserves_first_occurrence_and_metadata(self):
        df = pd.DataFrame({
            "text": ["Hello!", "Hello!"],
            "platform": ["reddit", "reddit"],
            "source_file": ["first.csv", "second.csv"],
        })

        result = remove_duplicate_posts(df)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["source_file"], "first.csv")

    def test_requires_text_column(self):
        df = pd.DataFrame({
            "message": ["Hello!"],
        })

        with self.assertRaisesRegex(ValueError, "text"):
            remove_duplicate_posts(df)

    def test_deduplicates_without_platform_column(self):
        df = pd.DataFrame({
            "text": ["Hello!", "Hello!", "Goodbye!"],
        })

        result = remove_duplicate_posts(df)

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()