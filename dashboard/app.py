from pathlib import Path
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_results.csv"

st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Social Media Sentiment Analysis")
st.write(
    "Explore sentiment predictions from your social media text analysis pipeline."
)


@st.cache_data
def load_data(path: str, modified_time: float) -> pd.DataFrame:
    """Load processed sentiment results."""
    return pd.read_csv(path)


if not DATA_PATH.exists():
    st.warning("No processed results found yet.")
    st.write("Run the pipeline from your project root first:")
    st.code("python -m src.pipeline", language="bash")
    st.stop()

df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)

required_columns = {"text", "label"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        "The processed CSV is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

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
    st.dataframe(platform_counts, width=True)

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
    width=True,
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
