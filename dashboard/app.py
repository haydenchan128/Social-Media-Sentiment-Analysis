from pathlib import Path
import pandas as pd
import streamlit as st
import json

from src.pipeline import run_multi_platform_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PIPELINE_OUTPUT_PATH = PROCESSED_DATA_DIR / "sentiment_results.csv"

PLATFORM_FILES = {
    "Reddit": RAW_DATA_DIR / "reddit_sample.csv",
    "TikTok": RAW_DATA_DIR / "tiktok_sample.csv",
    "Facebook": RAW_DATA_DIR / "facebook_sample.csv",
}

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_results.csv"

VADER_METRICS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "evaluation_report.json"
)
TRANSFORMER_METRICS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "transformer_metrics.json"
)
VADER_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "tweeteval_predictions.csv"
)
TRANSFORMER_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "transformer_predictions.csv"
)

st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide",
)

st.sidebar.header("Pipeline Settings")

# Only show platforms whose sample CSV files exist.
available_platforms = [
    platform
    for platform, file_path in PLATFORM_FILES.items()
    if file_path.exists()
]

selected_platforms = st.sidebar.multiselect(
    "Select platforms",
    options=available_platforms,
    default=available_platforms,
)

remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate posts",
    value=True,
)

selected_model = st.sidebar.selectbox(
    "Sentiment analysis model",
    options=["VADER", "Transformer"],
    index=0,
    help="Choose which model will analyze the selected posts.",
)

st.sidebar.divider()

if st.sidebar.button("Run Sentiment Analysis", type="primary"):
    if not selected_platforms:
        st.sidebar.warning("Select at least one platform.")
    else:
        input_files = {
            platform.lower(): str(PLATFORM_FILES[platform])
            for platform in selected_platforms
        }

        try:
            with st.spinner("Running sentiment analysis..."):
                results = run_multi_platform_pipeline(
                    input_files=input_files,
                    output_path=str(PIPELINE_OUTPUT_PATH),
                    remove_duplicates=remove_duplicates,
                    model=selected_model.lower(),
                )

            st.session_state["pipeline_message"] = (
                f"Successfully analyzed {len(results)} posts."
            )

            # Clear cached data so the dashboard loads the new results.
            st.cache_data.clear()
            st.rerun()

        except Exception as exc:
            st.sidebar.error(f"Pipeline failed: {exc}")

if "pipeline_message" in st.session_state:
    st.success(st.session_state.pop("pipeline_message"))

st.title("Social Media Sentiment Analysis")

analysis_tab, evaluation_tab = st.tabs([
    "Sentiment Analysis",
    "Model Evaluation",
])


@st.cache_data
def load_data(path: str, modified_time: float) -> pd.DataFrame:
    """Load processed sentiment results."""
    return pd.read_csv(path)


df = None

if DATA_PATH.exists():
    df = load_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        st.error(
            "The processed CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
        df = None


# Check whether the processed results match the selected model.
results_are_stale = False
results_model = None

if df is not None:
    if "model_used" not in df.columns:
        results_are_stale = True
    else:
        available_models = (
            df["model_used"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )

        if len(available_models) == 1:
            results_model = available_models[0]

        results_are_stale = (
            results_model != selected_model.lower()
        )


@st.cache_data
def load_metrics(path: str, modified_time: float) -> dict:
    """Load model evaluation metrics from a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

@st.cache_data
def load_predictions(path: str, modified_time: float) -> pd.DataFrame:
    """Load model predictions."""
    return pd.read_csv(path)


with analysis_tab:
    st.header("Sentiment Analysis")

    if df is None:
        st.info(
            "No processed sentiment results found. "
            "Run the pipeline using the sidebar settings to get started."
        )

    elif results_are_stale:
        st.warning(
            f"The existing results were generated using "
            f"{results_model.upper() if results_model else 'an unknown model'}, "
            f"but {selected_model} is currently selected. "
            "Run Sentiment Analysis again using the sidebar to generate "
            "updated results."
        )

    else:
        # Sidebar filters
        st.sidebar.divider()
        st.sidebar.header("Analysis Filters")

        filtered_df = df.copy()

        if "platform" in df.columns:
            platforms = sorted(df["platform"].dropna().unique().tolist())
            selected_platforms = st.sidebar.multiselect(
                "Platform",
                options=platforms,
                default=platforms,
            )
            filtered_df = filtered_df[
                filtered_df["platform"].isin(selected_platforms)
            ]

        sentiments = ["negative", "neutral", "positive"]
        available_labels = [
            label for label in sentiments
            if label in df["label"].dropna().unique()
        ]

        selected_sentiments = st.sidebar.multiselect(
            "Sentiment",
            options=available_labels,
            default=available_labels,
        )

        filtered_df = filtered_df[
            filtered_df["label"].isin(selected_sentiments)
        ]

        # Summary metrics
        st.subheader("Overview")

        total_posts = len(filtered_df)
        positive_pct = (
            filtered_df["label"].eq("positive").mean() * 100
            if total_posts else 0
        )
        neutral_pct = (
            filtered_df["label"].eq("neutral").mean() * 100
            if total_posts else 0
        )
        negative_pct = (
            filtered_df["label"].eq("negative").mean() * 100
            if total_posts else 0
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Posts analyzed", f"{total_posts:,}")
        col2.metric("Positive", f"{positive_pct:.1f}%")
        col3.metric("Neutral", f"{neutral_pct:.1f}%")
        col4.metric("Negative", f"{negative_pct:.1f}%")

        st.divider()

        # Sentiment distribution
        st.subheader("Sentiment Distribution")

        if filtered_df.empty:
            st.info("No posts match the selected filters.")
        else:
            counts = (
                filtered_df["label"]
                .value_counts()
                .reindex(sentiments, fill_value=0)
            )

            chart_data = counts.rename_axis("Sentiment").reset_index(name="Posts")
            st.bar_chart(chart_data, x="Sentiment", y="Posts")

        # Optional platform comparison
        if "platform" in filtered_df.columns:
            st.subheader("Sentiment by Platform")

            platform_counts = pd.crosstab(
                filtered_df["platform"],
                filtered_df["label"],
                normalize="index",
            ).mul(100).round(2)

            st.caption("Percentages within each platform.")
            st.dataframe(platform_counts, width="stretch")

        # Post explorer
        st.subheader("Explore Posts")

        display_columns = [
            column for column in [
                "text",
                "clean_text",
                "platform",
                "brand",
                "label",
                "compound",
                "positive",
                "neutral",
                "negative",
            ]
            if column in filtered_df.columns
        ]

        st.dataframe(
            filtered_df[display_columns],
            width="stretch",
            hide_index=True,
        )

        # Download filtered results
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download filtered results",
            data=csv_data,
            file_name="filtered_sentiment_results.csv",
            mime="text/csv",
        )


with evaluation_tab:
    st.header("Model Evaluation")

    st.write(
        "Compare VADER and Transformer using the TweetEval benchmark "
        "dataset. Explore model predictions, identify classification "
        "errors, and investigate disagreements between the models."
    )

    # Model Comparison
    st.divider()
    st.header("Model Comparison")

    if not VADER_METRICS_PATH.exists() or not TRANSFORMER_METRICS_PATH.exists():
        st.warning("Model evaluation files could not be found.")
    else:
        vader_metrics = load_metrics(
            str(VADER_METRICS_PATH),
            VADER_METRICS_PATH.stat().st_mtime,
        )

        transformer_metrics = load_metrics(
            str(TRANSFORMER_METRICS_PATH),
            TRANSFORMER_METRICS_PATH.stat().st_mtime,
        )

        vader_report = vader_metrics["classification_report"]
        transformer_report = transformer_metrics["classification_report"]

        # Accuracy comparison
        st.subheader("Overall Accuracy")

        accuracy_df = pd.DataFrame({
            "Model": ["VADER", "Transformer"],
            "Accuracy": [
                vader_metrics["accuracy"] * 100,
                transformer_metrics["accuracy"] * 100,
            ],
        })

        col1, col2 = st.columns(2)

        col1.metric(
            "VADER Accuracy",
            f"{vader_metrics['accuracy']:.2%}",
        )

        col2.metric(
            "Transformer Accuracy",
            f"{transformer_metrics['accuracy']:.2%}",
            delta=f"{(
                transformer_metrics['accuracy'] - vader_metrics['accuracy']
            ) * 100:.2f} percentage points",
        )

        st.bar_chart(
            accuracy_df,
            x="Model",
            y="Accuracy",
            y_label="Accuracy (%)",
        )

        # Per-class metric comparison
        st.subheader("Performance by Sentiment")

        selected_metric = st.selectbox(
            "Select evaluation metric",
            ["precision", "recall", "f1-score"],
            index=2,
        )

        sentiment_labels = ["negative", "neutral", "positive"]

        comparison_data = []

        for sentiment in sentiment_labels:
            comparison_data.append({
                "Sentiment": sentiment.capitalize(),
                "VADER": vader_report[sentiment][selected_metric] * 100,
                "Transformer": (
                    transformer_report[sentiment][selected_metric] * 100
                ),
            })

        comparison_df = pd.DataFrame(comparison_data)

        st.bar_chart(
            comparison_df,
            x="Sentiment",
            y=["VADER", "Transformer"],
            y_label=f"{selected_metric.capitalize()} (%)",
        )

        # Confusion matrices
        st.subheader("Confusion Matrices")

        vader_matrix_path = (
            PROJECT_ROOT / "data" / "processed" / "confusion_matrix.csv"
        )

        transformer_matrix_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "transformer_confusion_matrix.csv"
        )

        matrix_col1, matrix_col2 = st.columns(2)

        if vader_matrix_path.exists():
            vader_matrix = pd.read_csv(vader_matrix_path, index_col=0)
            matrix_col1.write("VADER")
            matrix_col1.dataframe(vader_matrix, width="stretch")

        if transformer_matrix_path.exists():
            transformer_matrix = pd.read_csv(
                transformer_matrix_path,
                index_col=0,
            )
            matrix_col2.write("Transformer")
            matrix_col2.dataframe(transformer_matrix, width="stretch")

        st.caption(
            "Both models were evaluated on the same 12,284-example "
            "TweetEval sentiment test split. Results may differ on other "
            "datasets and social media platforms."
        )

    # Model Prediction Explorer
    st.divider()
    st.header("Model Prediction Explorer")

    model_choice = st.selectbox(
        "Choose a model",
        ["VADER", "Transformer"],
    )

    if model_choice == "VADER":
        prediction_path = VADER_PREDICTIONS_PATH
    else:
        prediction_path = TRANSFORMER_PREDICTIONS_PATH

    if not prediction_path.exists():
        st.warning(f"{model_choice} prediction file not found.")
    else:
        predictions = load_predictions(
            str(prediction_path),
            prediction_path.stat().st_mtime,
        )

        required_columns = {
            "text",
            "true_label",
            "predicted_label",
        }

        if model_choice == "Transformer":
            required_columns.add("confidence")

        missing = required_columns - set(predictions.columns)

        if missing:
            st.error(
                "Missing columns: " + ", ".join(sorted(missing))
            )
        else:
            # Add a column showing whether the prediction was correct.
            predictions["correct"] = (
                predictions["true_label"]
                == predictions["predicted_label"]
            )

            # Sentiment filter
            available_sentiments = [
                label
                for label in ["negative", "neutral", "positive"]
                if label in predictions["predicted_label"].unique()
            ]

            selected_labels = st.multiselect(
                "Filter by predicted sentiment",
                options=available_sentiments,
                default=available_sentiments,
            )

            filtered_predictions = predictions[
                predictions["predicted_label"].isin(selected_labels)
            ].copy()

            # Summary metrics
            total = len(filtered_predictions)
            correct = filtered_predictions["correct"].sum()

            accuracy = (
                correct / total * 100
                if total > 0 else 0
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Posts", f"{total:,}")
            col2.metric("Correct predictions", f"{correct:,}")
            col3.metric("Accuracy", f"{accuracy:.2f}%")

            # Display predictions
            display_columns = [
                "text",
                "true_label",
                "predicted_label",
                "correct",
            ]

            if "confidence" in filtered_predictions.columns:
                filtered_predictions["confidence"] = (
                    filtered_predictions["confidence"].round(4)
                )
                display_columns.append("confidence")

            if "compound" in filtered_predictions.columns:
                filtered_predictions["compound"] = (
                    filtered_predictions["compound"].round(4)
                )
                display_columns.append("compound")

            st.subheader(f"{model_choice} Prediction Results")

            if model_choice == "Transformer":
                st.caption(
                    "Confidence represents the model's confidence in its predicted label. "
                    "It is not a measure of sentiment intensity."
                )
            else:
                st.caption(
                    "Compound scores represent the overall sentiment intensity, "
                    "ranging from -1 (negative) to +1 (positive)."
                )

            st.dataframe(
                filtered_predictions[display_columns],
                width="stretch",
                hide_index=True,
            )

            # Download predictions
            csv_data = filtered_predictions.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download model predictions",
                data=csv_data,
                file_name=f"{model_choice.lower()}_predictions.csv",
                mime="text/csv",
            )

    # Sentiment Error Analysis
    st.divider()
    st.header("Sentiment Error Analysis")

    if (
        VADER_PREDICTIONS_PATH.exists()
        and TRANSFORMER_PREDICTIONS_PATH.exists()
    ):
        vader_predictions = load_predictions(
            str(VADER_PREDICTIONS_PATH),
            VADER_PREDICTIONS_PATH.stat().st_mtime,
        )

        transformer_predictions = load_predictions(
            str(TRANSFORMER_PREDICTIONS_PATH),
            TRANSFORMER_PREDICTIONS_PATH.stat().st_mtime,
        )

        # Standardize prediction column names
        vader_errors = vader_predictions[
            ["text", "true_label", "predicted_label"]
        ].copy()

        transformer_errors = transformer_predictions[
            ["text", "true_label", "predicted_label"]
        ].copy()

        # Identify correct and incorrect predictions
        vader_errors["correct"] = (
            vader_errors["true_label"]
            == vader_errors["predicted_label"]
        )

        transformer_errors["correct"] = (
            transformer_errors["true_label"]
            == transformer_errors["predicted_label"]
        )

        # Overall error rates
        vader_error_rate = (
            (~vader_errors["correct"]).mean() * 100
        )

        transformer_error_rate = (
            (~transformer_errors["correct"]).mean() * 100
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "VADER Error Rate",
            f"{vader_error_rate:.2f}%",
        )

        col2.metric(
            "Transformer Error Rate",
            f"{transformer_error_rate:.2f}%",
            delta=f"{transformer_error_rate - vader_error_rate:.2f} percentage points",
            delta_color="inverse",
        )

        # Error breakdown by actual sentiment
        st.subheader("Errors by Actual Sentiment")

        error_comparison = []

        for sentiment in ["negative", "neutral", "positive"]:
            vader_subset = vader_errors[
                vader_errors["true_label"] == sentiment
            ]

            transformer_subset = transformer_errors[
                transformer_errors["true_label"] == sentiment
            ]

            error_comparison.append({
                "Sentiment": sentiment.capitalize(),
                "VADER": (
                    (~vader_subset["correct"]).mean() * 100
                ),
                "Transformer": (
                    (~transformer_subset["correct"]).mean() * 100
                ),
            })

        error_df = pd.DataFrame(error_comparison)

        st.bar_chart(
            error_df,
            x="Sentiment",
            y=["VADER", "Transformer"],
            y_label="Error Rate (%)",
        )

        # Inspect incorrect predictions
        st.subheader("Explore Misclassified Posts")

        selected_model = st.selectbox(
            "Choose model to inspect",
            ["VADER", "Transformer"],
            key="error_model",
        )

        selected_errors = (
            vader_errors
            if selected_model == "VADER"
            else transformer_errors
        )

        incorrect_posts = selected_errors[
            ~selected_errors["correct"]
        ]

        selected_sentiment = st.selectbox(
            "Filter by actual sentiment",
            ["All", "negative", "neutral", "positive"],
            key="error_sentiment",
        )

        if selected_sentiment != "All":
            incorrect_posts = incorrect_posts[
                incorrect_posts["true_label"] == selected_sentiment
            ]

        st.write(f"Misclassified posts: {len(incorrect_posts):,}")

        st.dataframe(
            incorrect_posts.drop(columns=["correct"]),
            width="stretch",
            hide_index=True,
        )

    else:
        st.warning("Prediction files are missing.")

    # Model Disagreement Analysis
    st.divider()
    st.header("Model Disagreement Analysis")

    if (
        VADER_PREDICTIONS_PATH.exists()
        and TRANSFORMER_PREDICTIONS_PATH.exists()
    ):
        vader_predictions = load_predictions(
            str(VADER_PREDICTIONS_PATH),
            VADER_PREDICTIONS_PATH.stat().st_mtime,
        )

        transformer_predictions = load_predictions(
            str(TRANSFORMER_PREDICTIONS_PATH),
            TRANSFORMER_PREDICTIONS_PATH.stat().st_mtime,
        )

        # Verify that both files contain the same test examples.
        same_examples = (
            len(vader_predictions) == len(transformer_predictions)
            and vader_predictions["text"].equals(
                transformer_predictions["text"]
            )
            and vader_predictions["true_label"].equals(
                transformer_predictions["true_label"]
            )
        )

        if not same_examples:
            st.error(
                "The prediction files do not contain matching examples "
                "in the same order. Cannot compare disagreements safely."
            )
        else:
            # Create a combined comparison dataframe.
            comparison = pd.DataFrame({
                "text": vader_predictions["text"],
                "true_label": vader_predictions["true_label"],
                "vader_prediction": vader_predictions["predicted_label"],
                "transformer_prediction": (
                    transformer_predictions["predicted_label"]
                ),
            })

            comparison["disagreement"] = (
                comparison["vader_prediction"]
                != comparison["transformer_prediction"]
            )

            disagreements = comparison[
                comparison["disagreement"]
            ].copy()

            agreement_count = len(comparison) - len(disagreements)

            agreement_rate = (
                agreement_count / len(comparison) * 100
                if len(comparison) > 0 else 0
            )

            disagreement_rate = (
                len(disagreements) / len(comparison) * 100
                if len(comparison) > 0 else 0
            )

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Posts",
                f"{len(comparison):,}",
            )

            col2.metric(
                "Agreement Rate",
                f"{agreement_rate:.2f}%",
            )

            col3.metric(
                "Disagreement Rate",
                f"{disagreement_rate:.2f}%",
            )

            # Disagreement breakdown
            st.subheader("Disagreements by Sentiment")

            disagreement_counts = (
                disagreements.groupby("true_label")
                .size()
                .reindex(
                    ["negative", "neutral", "positive"],
                    fill_value=0,
                )
                .rename_axis("Actual Sentiment")
                .reset_index(name="Disagreements")
            )

            st.bar_chart(
                disagreement_counts,
                x="Actual Sentiment",
                y="Disagreements",
            )

            # Filter disagreements
            st.subheader("Explore Disagreements")

            selected_actual = st.selectbox(
                "Filter by actual sentiment",
                ["All", "negative", "neutral", "positive"],
                key="disagreement_sentiment",
            )

            if selected_actual != "All":
                disagreements = disagreements[
                    disagreements["true_label"] == selected_actual
                ]

            st.write(
                f"Showing {len(disagreements):,} disagreements."
            )

            st.dataframe(
                disagreements[
                    [
                        "text",
                        "true_label",
                        "vader_prediction",
                        "transformer_prediction",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

            # Download disagreements
            csv_data = disagreements.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download disagreements",
                data=csv_data,
                file_name="model_disagreements.csv",
                mime="text/csv",
            )

    else:
        st.warning("Prediction files are missing.")