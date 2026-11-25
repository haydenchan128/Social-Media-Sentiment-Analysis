import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src.pipeline import run_pipeline, run_multi_platform_pipeline


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_single_platform_pipeline(self):
        input_path = self.test_dir / "reddit.csv"
        output_path = self.test_dir / "results.csv"

        pd.DataFrame({
            "text": [
                "I love this!",
                "This is terrible."
            ]
        }).to_csv(input_path, index=False)

        result = run_pipeline(str(input_path), str(output_path))

        self.assertEqual(len(result), 2)
        self.assertIn("label", result.columns)
        self.assertTrue(output_path.exists())

    def test_multi_platform_pipeline(self):
        reddit_path = self.test_dir / "reddit.csv"
        tiktok_path = self.test_dir / "tiktok.csv"
        output_path = self.test_dir / "results.csv"

        pd.DataFrame({
            "text": ["I love this!"]
        }).to_csv(reddit_path, index=False)

        pd.DataFrame({
            "text": ["This is awful."]
        }).to_csv(tiktok_path, index=False)

        input_files = {
            "reddit": str(reddit_path),
            "tiktok": str(tiktok_path)
        }

        result = run_multi_platform_pipeline(
            input_files,
            str(output_path)
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            set(result["platform"]),
            {"reddit", "tiktok"}
        )
        self.assertIn("label", result.columns)
        self.assertTrue(output_path.exists())

    def test_empty_input_files(self):
        output_path = self.test_dir / "results.csv"

        with self.assertRaises(ValueError):
            run_multi_platform_pipeline({}, str(output_path))


if __name__ == "__main__":
    unittest.main()
