# Social-Media-Sentiment-Analysis

## Baseline Model Evaluation

The project uses VADER, a rule-based sentiment analysis tool, as its baseline model. It was evaluated on the test split of the TweetEval sentiment dataset.

### Results
- **Accuracy**: 52.93%
- **Negative F1 score**: 0.5816
- **Neutral F1 score**: 0.5290
- **Positive F1 score**: 0.4738

### Analysis

VADER performed best on negative sentiment based on F1 score. Its weakest F1 score was for positive sentiment. The confusion matrix also shows substantial confusion between neutral and positive tweets.

These results demonstrate the limitations of a rule-based sentiment analyzer on social media text. Slang, sarcasm, context, and differences between annotation standards and VADER's sentiment rules may contribute to classification errors.