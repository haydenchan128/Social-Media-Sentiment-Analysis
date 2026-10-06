import re


def clean_text(text: str) -> str:
    """
    Clean social media text while preserving information
    that may be useful for sentiment analysis.
    """

    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Replace usernames with a generic token
    text = re.sub(r"@\w+", "@USER", text)

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()