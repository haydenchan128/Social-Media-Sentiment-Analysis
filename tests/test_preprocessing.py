import unittest

from src.preprocessing import clean_text


class TestPreprocessing(unittest.TestCase):

    def test_removes_urls(self):
        result = clean_text("Check this out https://example.com")
        self.assertEqual(result, "Check this out")

    def test_normalizes_mentions(self):
        result = clean_text("Hello @john")
        self.assertEqual(result, "Hello @USER")

    def test_preserves_hashtags(self):
        result = clean_text("I love this! #Amazing")
        self.assertEqual(result, "I love this! #Amazing")

    def test_preserves_emojis(self):
        result = clean_text("I love this! 😊")
        self.assertEqual(result, "I love this! 😊")

    def test_preserves_punctuation(self):
        result = clean_text("This is amazing!!!")
        self.assertEqual(result, "This is amazing!!!")

    def test_preserves_negations(self):
        result = clean_text("I do not like this.")
        self.assertEqual(result, "I do not like this.")

    def test_removes_extra_whitespace(self):
        result = clean_text("  Hello    world!  ")
        self.assertEqual(result, "Hello world!")

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")

    def test_whitespace_only(self):
        self.assertEqual(clean_text("   "), "")

    def test_non_string_input(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")


if __name__ == "__main__":
    unittest.main()
