import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai

# ==================================================
# GEMINI CONFIG (PRIVATE REPO ONLY)
# ⚠️ DO NOT COMMIT THIS KEY IF REPO BECOMES PUBLIC
# ==================================================
GEMINI_API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Aadhaar Trend Intelligence Copilot (ATIC)",
    layout="wide"
)

st.title("🆔 Aadhaar Trend Intelligence Copilot (ATIC)")
st.caption(
    "Read-only analytics & AI explanations derived exclusively "
    "from pre-computed Aadhaar stress indicators"
)

# ==================================================
# LOAD DATA (SINGLE SOURCE OF TRUTH)
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("district_stress_index.csv")

df = load_data()

st.success(
    f"Loaded {len(df):,} district-month records "
    "from district_stress_index.csv"
)

# ==================================================
# REQUIRED COLUMN VALIDATION (FAIL FAST)
# ==================================================
required_cols = {
    "state", "district", "month", "DSI", "stress_level"
}
missing = required_cols - set(df.columns)

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ==================================================
# POLICY INTERPRETATION (PRESENTATION ONLY)
# ==================================================
ACTION_MAP = {
    "Extreme": "Immediate audit and targeted enrolment/update drive",
    "High": "Closer monitoring and operational review",
    "Moderate": "Routine monitoring",
    "Low": "No action required"
}

if "recommended_action" not in df.columns:
    df["recommended_action"] = df["stress_level"].map(ACTION_MAP)

if "priority_level" not in df.columns:
    df["priority_level"] = df["stress_level"].map({
        "Extreme": "High Priority",
        "High": "High Priority",
        "Moderate": "Medium Priority",
        "Low": "Low Priority"
    })

# ==================================================
# FILTERS
# ==================================================
st.subheader("🔎 Filters")

c1, c2 = st.columns(2)

with c1:
    states = st.multiselect(
        "Select State(s)",
        sorted(df["state"].unique())
    )

with c2:
    months = st.multiselect(
        "Select Month(s)",
        sorted(df["month"].unique())
    )

filtered_df = df.copy()
if states:
    filtered_df = filtered_df[filtered_df["state"].isin(states)]
if months:
    filtered_df = filtered_df[filtered_df["month"].isin(months)]

# ==================================================
# OUTPUT 1 — TOP STRESSED DISTRICTS
# ==================================================
st.subheader("🔴 Top Stressed Districts")

st.dataframe(
    filtered_df
    .sort_values("DSI", ascending=False)
    .head(10)[[
        "state",
        "district",
        "DSI",
        "stress_level",
        "recommended_action"
    ]],
    use_container_width=True
)

# ==================================================
# OUTPUT 2 — DSI TREND
# ==================================================
st.subheader("📈 Average District Stress Index (DSI) Over Time")

fig1, ax1 = plt.subplots(figsize=(10, 4))
filtered_df.groupby("month")["DSI"].mean().plot(
    ax=ax1,
    marker="o"
)
ax1.set_xlabel("Month")
ax1.set_ylabel("Average DSI")
ax1.set_title("DSI Trend (Loaded from Saved Output)")
st.pyplot(fig1)

# ==================================================
# OUTPUT 3 — STRESS DISTRIBUTION (LATEST MONTH)
# ==================================================
st.subheader("📊 District Stress Level Distribution")

latest_month = filtered_df["month"].max()

dist = (
    filtered_df[filtered_df["month"] == latest_month]["stress_level"]
    .value_counts()
    .sort_index()
)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(dist.index, dist.values)
ax2.set_xlabel("Stress Level")
ax2.set_ylabel("Number of Districts")
ax2.set_title(f"Stress Distribution — {latest_month}")
st.pyplot(fig2)

# ==================================================
# GEMINI EXPLANATION LAYER (READ-ONLY)
# ==================================================
st.subheader("💬 Ask the Copilot")

query = st.text_area(
    "Ask about trends, districts, stress causes, or actions",
    placeholder="Why are some districts extremely stressed this month?"
)

if query:
    with st.spinner("Analysing using Gemini…"):
        context = f"""
You are an Aadhaar policy analytics assistant.

IMPORTANT RULES:
- All numbers are pre-computed and loaded from CSV
- Do NOT invent data
- Do NOT speculate beyond the data

Dataset summary:
- Records: {len(filtered_df)}
- Average DSI: {round(filtered_df['DSI'].mean(), 3)}
- Maximum DSI: {round(filtered_df['DSI'].max(), 3)}
- Stress distribution: {filtered_df['stress_level'].value_counts().to_dict()}

Top stressed districts:
{filtered_df.sort_values("DSI", ascending=False)
.head(5)[['state','district','DSI','stress_level']]
.to_string(index=False)}

User question:
{query}

Explain clearly for government decision-makers.
"""

        response = model.generate_content(context)
        st.markdown("### 🧠 Copilot Explanation")
        st.write(response.text)

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "ATIC • UIDAI–NIC Hackathon • "
    "All analytics loaded from district_stress_index.csv • "
    "LLM used strictly for explanation"
)
