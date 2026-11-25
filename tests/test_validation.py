
import unittest
import pandas as pd

from src.validation import validate_input_data


class TestValidation(unittest.TestCase):

    def test_valid_data(self):
        df = pd.DataFrame({
            "text": ["I love this!", "This is terrible."]
        })

        result = validate_input_data(df)

        self.assertEqual(len(result), 2)
        self.assertIsNot(result, df)

    def test_empty_dataset(self):
        df = pd.DataFrame(columns=["text"])

        with self.assertRaises(ValueError):
            validate_input_data(df)

    def test_missing_text_column(self):
        df = pd.DataFrame({
            "message": ["Hello"]
        })

        with self.assertRaises(ValueError):
            validate_input_data(df)

    def test_all_missing_text(self):
        df = pd.DataFrame({
            "text": [None, None]
        })

        with self.assertRaises(ValueError):
            validate_input_data(df)

    def test_non_string_text(self):
        df = pd.DataFrame({
            "text": ["Hello", 123]
        })

        with self.assertRaises(ValueError):
            validate_input_data(df)

    def test_all_blank_text(self):
        df = pd.DataFrame({
            "text": ["  ", ""]
        })

        with self.assertRaises(ValueError):
            validate_input_data(df)


if __name__ == "__main__":
    unittest.main()
