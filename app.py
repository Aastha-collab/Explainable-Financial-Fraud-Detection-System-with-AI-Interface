# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Explainable Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# PREMIUM MODERN DARK-LUXURY UI CSS
# REPLACE YOUR OLD CSS SECTION WITH THIS
# =========================================================

st.markdown("""
<style>

/* MAIN APP */

.stApp {

    background: linear-gradient(
        135deg,
        #0F172A 0%,
        #111827 40%,
        #1E1B4B 100%
    );

    color: #F8FAFC;

    font-family: 'Segoe UI', sans-serif;
}

/* HIDE DEFAULT STREAMLIT */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* MAIN TITLES */

h1 {

    color: #F9FAFB !important;

    font-size: 3rem !important;

    font-weight: 800 !important;

    letter-spacing: 1px;
}

h2 {

    color: #C084FC !important;

    font-weight: 700 !important;
}

h3 {

    color: #F9A8D4 !important;

    font-weight: 700 !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #111827 0%,
        #1E293B 100%
    );

    border-right: 2px solid #312E81;
}

/* SIDEBAR TEXT */

[data-testid="stSidebar"] * {

    color: #F8FAFC !important;
}

/* DROPDOWN */

[data-testid="stSidebar"] .stSelectbox > div > div {

    background: #1E293B !important;

    color: white !important;

    border-radius: 14px !important;

    border: 2px solid #8B5CF6 !important;

    box-shadow: 0px 2px 12px rgba(139,92,246,0.2);
}

/* DROPDOWN TEXT */

[data-testid="stSidebar"] .stSelectbox * {

    color: white !important;

    font-weight: 600 !important;
}

/* DROPDOWN ARROW */

[data-testid="stSidebar"] svg {

    fill: #C084FC !important;
}

/* METRIC CARDS */

.metric-card {

    background: linear-gradient(
        135deg,
        #1E293B 0%,
        #0F172A 100%
    );

    padding: 24px;

    border-radius: 24px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
    0px 8px 24px rgba(0,0,0,0.35);

    text-align: center;

    transition: 0.3s;
}

.metric-card:hover {

    transform: translateY(-4px);

    box-shadow:
    0px 12px 28px rgba(139,92,246,0.25);
}

/* METRIC VALUE */

.metric-value {

    font-size: 42px;

    font-weight: 800;

    color: #F9FAFB;
}

/* METRIC LABEL */

.metric-label {

    color: #CBD5E1;

    font-size: 16px;

    margin-bottom: 10px;
}

/* RESULT BOX */

.result-box {

    background: linear-gradient(
        135deg,
        #1E293B 0%,
        #111827 100%
    );

    padding: 28px;

    border-radius: 24px;

    border-left: 8px solid #EC4899;

    box-shadow:
    0px 6px 22px rgba(0,0,0,0.3);

    margin-top: 15px;
}

/* AI BOX */

.ai-box {

    background: linear-gradient(
        135deg,
        #312E81 0%,
        #1E1B4B 100%
    );

    padding: 28px;

    border-radius: 24px;

    border-left: 8px solid #A855F7;

    box-shadow:
    0px 6px 22px rgba(168,85,247,0.25);

    color: #F8FAFC;

    line-height: 1.8;

    font-size: 16px;
}

/* BUTTON */

.stButton>button {

    background: linear-gradient(
        to right,
        #8B5CF6,
        #EC4899
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 16px !important;

    font-size: 16px !important;

    font-weight: 700 !important;

    padding: 12px 18px !important;

    width: 100% !important;

    transition: 0.3s !important;

    box-shadow:
    0px 6px 18px rgba(236,72,153,0.25);
}

.stButton>button:hover {

    transform: scale(1.02);

    background: linear-gradient(
        to right,
        #7C3AED,
        #DB2777
    ) !important;
}

/* INPUT BOX */

.stTextInput input {

    background: #111827 !important;

    color: white !important;

    border-radius: 12px !important;

    border: 2px solid #8B5CF6 !important;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {

    background: #111827 !important;

    border-radius: 18px !important;

    padding: 12px !important;

    border: 1px solid rgba(255,255,255,0.08);
}

/* DATAFRAME TEXT */

[data-testid="stDataFrame"] * {

    color: white !important;
}

/* EXPANDER */

.streamlit-expanderHeader {

    background: #1E293B !important;

    border-radius: 12px !important;

    color: white !important;
}

/* EXPANDER CONTENT */

.streamlit-expanderContent {

    background: #0F172A !important;

    border-radius: 12px !important;

    color: white !important;
}

/* CHATBOT BOX */

.chat-box {

    background: linear-gradient(
        135deg,
        #1E293B 0%,
        #0F172A 100%
    );

    padding: 24px;

    border-radius: 20px;

    border-left: 8px solid #06B6D4;

    box-shadow:
    0px 6px 20px rgba(6,182,212,0.2);

    color: #F8FAFC;

    line-height: 1.8;
}

/* BLOCK CONTAINER */

.block-container {

    padding-top: 2rem;
}

/* SUCCESS BOX */

.stSuccess {

    border-radius: 16px !important;
}

/* ERROR BOX */

.stError {

    border-radius: 16px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
# 💳 Explainable Financial Fraud Detection System

### AI-Based Fraud Detection, Risk Analysis & Chatbot Intelligence Dashboard

This system detects suspicious financial transactions using Machine Learning and explains the fraud reasoning through intelligent risk analysis and chatbot interaction.
""")

# =========================================================
# LOAD FILES
# =========================================================

rf_model = joblib.load("rf_model.pkl")
top_features = joblib.load("top_features.pkl")
normal_means = joblib.load("normal_means.pkl")

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("creditcard_sample.csv")
df['Hour'] = df['Time'] // 3600

X = df.drop("Class", axis=1)
y = df["Class"]

fraud_df = df[df["Class"] == 1]
normal_df = df[df["Class"] == 0]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("# 🔍 Transaction Explorer")

st.sidebar.markdown("""
Choose a transaction to:

✔ Detect Fraud  
✔ Analyze Risk Factors  
✔ Generate Explanation  
✔ Interact with AI Chatbot
""")

index = st.sidebar.selectbox(
    "Choose Transaction Row",
    options=df.index.tolist()
)

transaction = X.iloc[index]

# =========================================================
# PREDICTION
# =========================================================

prediction = rf_model.predict(
    transaction.to_frame().T
)[0]

proba = rf_model.predict_proba(
    transaction.to_frame().T
)[0][1]

# =========================================================
# RISK FACTOR FUNCTION
# =========================================================

def get_risk_factors(
    transaction,
    top_features,
    normal_means,
    threshold=2
):

    risk_factors = []

    for feature in top_features:

        value = transaction[feature]

        mean_value = normal_means[feature]

        if abs(value - mean_value) > threshold:

            risk_factors.append(feature)

    return risk_factors

# =========================================================
# EXPLANATION FUNCTION
# =========================================================

def generate_explanation(
    prediction,
    risk_factors
):

    if prediction == 0:

        if len(risk_factors) == 0:

            return "Transaction appears normal with stable behavior patterns."

        else:

            return f"Transaction appears mostly normal but shows slight deviations in {', '.join(risk_factors)}."

    else:

        if len(risk_factors) == 0:

            return "Transaction flagged suspicious due to anomaly detection."

        else:

            return f"Transaction flagged suspicious due to abnormal behavior in {', '.join(risk_factors)}."

# =========================================================
# RESULTS
# =========================================================

risk_factors = get_risk_factors(
    transaction,
    top_features,
    normal_means
)

explanation = generate_explanation(
    prediction,
    risk_factors
)

# =========================================================
# ANALYTICS
# =========================================================

st.markdown("## 📊 System Analytics")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(f"""
    <div class="metric-card">
    <h3>Total Transactions</h3>
    <h1>{len(df)}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="metric-card">
    <h3>Fraud Cases</h3>
    <h1>{len(fraud_df)}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="metric-card">
    <h3>Normal Cases</h3>
    <h1>{len(normal_df)}</h1>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREDICTION RESULT
# =========================================================

st.markdown("## 🎯 Prediction Result")

if prediction == 1:

    st.error("🚨 Fraud Transaction Detected")

else:

    st.success("✅ Normal Transaction")

# =========================================================
# PROBABILITY CARD
# =========================================================

st.markdown(f"""
<div class="chat-box">

<h3>📌 Fraud Probability</h3>

<h2>{round(proba*100,2)}%</h2>

</div>
""", unsafe_allow_html=True)

# =========================================================
# RISK FACTORS
# =========================================================

st.markdown("## ⚠️ Risk Factors")

if len(risk_factors) > 0:

    st.markdown(f"""
    <div class="chat-box">

    <b>Detected Risk Features:</b><br><br>

    {", ".join(risk_factors)}

    </div>
    """, unsafe_allow_html=True)

else:

    st.success("No major risk factors detected.")

# =========================================================
# EXPLANATION
# =========================================================

st.markdown("## 🤖 AI Explanation")

st.markdown(f"""
<div class="chat-box">

{explanation}

</div>
""", unsafe_allow_html=True)

# =========================================================
# TRANSACTION DETAILS
# =========================================================

with st.expander("📄 View Transaction Details"):

    st.dataframe(
        transaction.to_frame().T,
        use_container_width=True
    )

# =========================================================
# BAR GRAPH
# =========================================================

st.markdown("## 📊 Feature Visualization")

selected_features = risk_factors[:5]

if len(selected_features) == 0:

    selected_features = top_features[:5]

values = [
    abs(transaction[feature])
    for feature in selected_features
]

fig, ax = plt.subplots(figsize=(6,3))

colors = [
    '#EC4899',
    '#A855F7',
    '#6366F1',
    '#06B6D4',
    '#10B981'
]

ax.bar(
    selected_features,
    values,
    color=colors[:len(selected_features)]
)

ax.set_title(
    "Risk Feature Scores",
    fontsize=12
)

ax.tick_params(
    axis='x',
    labelsize=9
)

ax.tick_params(
    axis='y',
    labelsize=9
)

fig.patch.set_facecolor('#F8FAFF')

ax.set_facecolor('#FFFFFF')

st.pyplot(fig)

# =========================================================
# PIE CHART
# =========================================================

st.markdown("## 📈 Dataset Distribution")

fig2, ax2 = plt.subplots(figsize=(4,4))

colors = ['#A78BFA', '#F472B6']

ax2.pie(
    [
        len(normal_df),
        len(fraud_df)
    ],
    labels=['Normal', 'Fraud'],
    autopct='%1.1f%%',
    colors=colors,
    textprops={'fontsize': 10}
)

fig2.patch.set_facecolor('#F8FAFF')

st.pyplot(fig2)

# =========================================================
# CHATBOT
# =========================================================

st.markdown("## 💬 AI Fraud Chatbot")

user_query = st.text_input(
    "Ask about prediction, risk, fraud reason, or model"
)

def chatbot(query):

    q = query.lower()

    if "prediction" in q:

        return "Fraud Transaction" if prediction == 1 else "Normal Transaction"

    elif "why" in q:

        return explanation

    elif "risk" in q:

        return f"Risk Factors: {risk_factors}"

    elif "model" in q:

        return "The model used is Random Forest Classifier."

    else:

        return "Ask about fraud prediction, risk factors, explanation, or model."

# =========================================================
# CHATBOT OUTPUT
# =========================================================

if user_query:

    response = chatbot(user_query)

    st.markdown(f"""
    <div class="chat-box">

    <h3>🤖 AI Assistant Response</h3>

    {response}

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# BUSINESS IMPACT
# =========================================================

st.markdown("""
---

## 💼 Business Impact

This system helps financial institutions to:

- Detect suspicious transactions faster
- Improve fraud explainability
- Understand transaction risk behavior
- Reduce manual fraud investigation time
- Enhance AI-driven financial monitoring systems
""")
