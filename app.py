import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("real_estate_buyer_segments.csv")
    return df

df = load_data()

# -----------------------------
# TITLE
# -----------------------------
st.title("🏠 Real Estate Buyer Segmentation & Investment Profiling")
st.markdown(
    "### Machine Learning Based Buyer Segmentation and Investment Profiling "
    "for Real Estate Market Intelligence"
)

st.divider()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔎 Filters")

countries = sorted(df["country"].dropna().unique())
regions = sorted(df["region"].dropna().unique())
purposes = sorted(df["acquisition_purpose"].dropna().unique())
client_types = sorted(df["client_type"].dropna().unique())

selected_country = st.sidebar.multiselect(
    "Country",
    countries
)

selected_region = st.sidebar.multiselect(
    "Region",
    regions
)

selected_purpose = st.sidebar.multiselect(
    "Acquisition Purpose",
    purposes
)

selected_client = st.sidebar.multiselect(
    "Client Type",
    client_types
)

filtered_df = df.copy()

if selected_country:
    filtered_df = filtered_df[
        filtered_df["country"].isin(selected_country)
    ]

if selected_region:
    filtered_df = filtered_df[
        filtered_df["region"].isin(selected_region)
    ]

if selected_purpose:
    filtered_df = filtered_df[
        filtered_df["acquisition_purpose"].isin(selected_purpose)
    ]

if selected_client:
    filtered_df = filtered_df[
        filtered_df["client_type"].isin(selected_client)
    ]

# -----------------------------
# SECTION 1
# BUYER SEGMENTATION OVERVIEW
# -----------------------------
st.header("1️⃣ Buyer Segmentation Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Buyers",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Buyer Segments",
        filtered_df["buyer_segment"].nunique()
    )

with col3:
    st.metric(
        "Average Age",
        f"{filtered_df['age'].mean():.1f}"
    )

with col4:
    st.metric(
        "Average Investment",
        f"₹{filtered_df['investment_amount'].mean()/100000:.1f} L"
    )

st.subheader("Buyer Segment Distribution")

segment_counts = (
    filtered_df["buyer_segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["buyer_segment", "count"]

fig1 = px.bar(
    segment_counts,
    x="buyer_segment",
    y="count",
    title="Number of Buyers by Segment",
    text="count"
)

fig1.update_layout(
    xaxis_title="Buyer Segment",
    yaxis_title="Number of Buyers"
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# SECTION 2
# INVESTOR BEHAVIOR DASHBOARD
# -----------------------------
st.header("2️⃣ Investor Behavior Dashboard")

col1, col2 = st.columns(2)

with col1:
    purpose_counts = (
        filtered_df["acquisition_purpose"]
        .value_counts()
        .reset_index()
    )

    purpose_counts.columns = ["purpose", "count"]

    fig2 = px.pie(
        purpose_counts,
        names="purpose",
        values="count",
        title="Acquisition Purpose"
    )

    st.plotly_chart(fig2, use_container_width=True)

with col2:
    loan_counts = (
        filtered_df["loan_applied"]
        .value_counts()
        .reset_index()
    )

    loan_counts.columns = ["loan_applied", "count"]

    fig3 = px.bar(
        loan_counts,
        x="loan_applied",
        y="count",
        title="Loan Application Behavior",
        text="count"
    )

    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Average Investment by Buyer Segment")

investment_segment = (
    filtered_df.groupby("buyer_segment")["investment_amount"]
    .mean()
    .reset_index()
)

fig4 = px.bar(
    investment_segment,
    x="buyer_segment",
    y="investment_amount",
    title="Average Investment by Segment",
    text_auto=".2s"
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# SECTION 3
# GEOGRAPHIC BUYER ANALYSIS
# -----------------------------
st.header("3️⃣ Geographic Buyer Analysis")

col1, col2 = st.columns(2)

with col1:
    country_counts = (
        filtered_df["country"]
        .value_counts()
        .reset_index()
    )

    country_counts.columns = ["country", "count"]

    fig5 = px.bar(
        country_counts,
        x="country",
        y="count",
        title="Buyers by Country",
        text="count"
    )

    st.plotly_chart(fig5, use_container_width=True)

with col2:
    region_counts = (
        filtered_df["region"]
        .value_counts()
        .reset_index()
    )

    region_counts.columns = ["region", "count"]

    fig6 = px.bar(
        region_counts,
        x="region",
        y="count",
        title="Buyers by Region",
        text="count"
    )

    st.plotly_chart(fig6, use_container_width=True)

st.subheader("Buyer Segments by Country")

country_segment = (
    filtered_df.groupby(
        ["country", "buyer_segment"]
    )
    .size()
    .reset_index(name="count")
)

fig7 = px.bar(
    country_segment,
    x="country",
    y="count",
    color="buyer_segment",
    title="Buyer Segments Across Countries"
)

st.plotly_chart(fig7, use_container_width=True)

# -----------------------------
# SECTION 4
# SEGMENT INSIGHTS PANEL
# -----------------------------
st.header("4️⃣ Segment Insights Panel")

segments = sorted(
    filtered_df["buyer_segment"].dropna().unique()
)

selected_segment = st.selectbox(
    "Select a Buyer Segment",
    segments
)

segment_df = filtered_df[
    filtered_df["buyer_segment"] == selected_segment
]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Buyers",
        len(segment_df)
    )

with col2:
    st.metric(
        "Average Age",
        f"{segment_df['age'].mean():.1f}"
    )

with col3:
    st.metric(
        "Average Income",
        f"₹{segment_df['annual_income'].mean()/100000:.1f} L"
    )

with col4:
    st.metric(
        "Average Investment",
        f"₹{segment_df['investment_amount'].mean()/100000:.1f} L"
    )

col1, col2 = st.columns(2)

with col1:
    st.write("### 📊 Property Budget")

    st.metric(
        "Average Property Budget",
        f"₹{segment_df['property_budget'].mean()/10000000:.2f} Cr"
    )

    st.write("### ⭐ Satisfaction")

    st.metric(
        "Average Satisfaction",
        f"{segment_df['satisfaction_score'].mean():.2f}/5"
    )

with col2:
    st.write("### 🌍 Top Country")

    top_country = segment_df["country"].mode()

    if not top_country.empty:
        st.info(top_country.iloc[0])

    st.write("### 📍 Top Region")

    top_region = segment_df["region"].mode()

    if not top_region.empty:
        st.info(top_region.iloc[0])

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.divider()

with st.expander("📋 View Filtered Dataset"):
    st.dataframe(
        filtered_df,
        use_container_width=True
    )

st.success(
    "Dashboard successfully loaded — Real Estate Buyer Intelligence"
)
