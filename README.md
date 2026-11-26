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


## Transformer Model Evaluation

A pretrained Twitter-RoBERTa sentiment model was evaluated on the same TweetEval sentiment test split as the VADER baseline.

### Results

- **Accuracy:** 72.17%
- **Negative F1 Score:** 0.7435
- **Neutral F1 Score:** 0.7030
- **Positive F1 Score:** 0.7255

### Comparison with VADER

The transformer achieved 72.17% accuracy, compared with 52.93% for VADER, an improvement of 19.24 percentage points. It also achieved higher F1 scores across all three sentiment classes.

The largest improvement was in positive sentiment, suggesting that the transformer handles this category more effectively on this test dataset.

### Conclusion

On the TweetEval test split, the transformer outperformed the rule-based VADER baseline in accuracy and per-class F1 scores. This comparison demonstrates the potential benefits of pretrained transformer models for social media sentiment analysis.

These results apply to the evaluated dataset and may not generalize to other platforms or datasets. Because the transformer was pretrained and fine-tuned for Twitter sentiment, its performance should be interpreted in that context.


## Running the Dashboard with Docker

This project supports Docker, allowing you to run the Streamlit dashboard in an isolated environment without installing the project's Python dependencies locally.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Build the Docker Image

From the project's root directory, run:

```bash
docker build -t sentiment-analysis .
```

### Run the Dashboard

```bash
docker run -p 8501:8501 \
  -v "$(pwd)/data/processed:/app/data/processed:ro" \
  sentiment-analysis
```

### Access the Dashboard

Open the following URL in your browser:

http://localhost:8501

To stop the dashboard, press `Ctrl + C` in the terminal running the container.