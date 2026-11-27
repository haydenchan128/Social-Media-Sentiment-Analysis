import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src.pipeline import run_pipeline, run_multi_platform_pipeline, analyze_dataframe


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

    def test_run_pipeline_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "missing.csv"
            output_path = Path(temp_dir) / "output.csv"

            with self.assertRaises(FileNotFoundError):
                run_pipeline(input_path, output_path)


    def test_run_pipeline_empty_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "empty.csv"
            output_path = Path(temp_dir) / "output.csv"

            input_path.write_text("")

            with self.assertRaises(ValueError):
                run_pipeline(input_path, output_path)


    def test_run_pipeline_malformed_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "malformed.csv"
            output_path = Path(temp_dir) / "output.csv"

            input_path.write_text('text,platform\n"Unclosed quote,Reddit\n')

            with self.assertRaisesRegex(ValueError, "Malformed CSV"):
                run_pipeline(input_path, output_path)

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

    def test_failed_pipeline_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "invalid.csv"
            output_path = Path(temp_dir) / "output" / "results.csv"

            input_path.write_text("message\nHello world\n")

            with self.assertRaisesRegex(ValueError, "text"):
                run_pipeline(input_path, output_path)

            self.assertFalse(output_path.exists())    

    def test_pipeline_preserves_platform_metadata(self):
        df = pd.DataFrame({
            "text": ["I love this!"],
            "platform": ["reddit"],
            "source_file": ["reddit_sample.csv"]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.csv"

            result = analyze_dataframe(df, output_path)

            self.assertEqual(result["platform"].iloc[0], "reddit")
            self.assertEqual(
                result["source_file"].iloc[0],
                "reddit_sample.csv"
            )
            self.assertTrue(output_path.exists())

    def test_pipeline_removes_duplicate_posts(self):
        df = pd.DataFrame({
            "text": [
                "I love this!",
                "I love this!",
                "This is terrible!",
            ],
            "platform": [
                "reddit",
                "reddit",
                "reddit",
            ],
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.csv"

            result = analyze_dataframe(df, str(output_path))

            self.assertEqual(len(result), 2)
            self.assertTrue(output_path.exists())

    def test_pipeline_can_keep_duplicates(self):
        df = pd.DataFrame({
            "text": [
                "I love this!",
                "I love this!",
                "This is terrible!",
            ],
            "platform": ["reddit", "reddit", "reddit"],
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.csv"

            result = analyze_dataframe(
                df,
                str(output_path),
                remove_duplicates=False,
            )

            self.assertEqual(len(result), 3)

    def test_pipeline_rejects_unsupported_model(self):
        df = pd.DataFrame({
            "text": ["This is a test."],
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.csv"

            with self.assertRaisesRegex(ValueError, "Unsupported sentiment model"):
                analyze_dataframe(
                    df,
                    str(output_path),
                    model="invalid",
                )

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
