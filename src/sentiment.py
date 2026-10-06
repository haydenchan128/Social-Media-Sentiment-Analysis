from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        """Analyze text sentiment using VADER."""

        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        label = ("positive" if compound >= 0.05
            else ("negative" if compound <= -0.05 else "neutral")
        )

        return {
            "label": label,
            "compound": compound,
            "positive": scores["pos"],
            "neutral": scores["neu"],
            "negative": scores["neg"],
        }