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
# PREMIUM UI CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #F9FAFF 0%,
        #EEF2FF 50%,
        #FDF2F8 100%
    );
    color: #1E293B;
    font-family: 'Segoe UI', sans-serif;
}

/* HIDE STREAMLIT DEFAULTS */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* HEADINGS */

h1 {
    color: #6D28D9 !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #7C3AED !important;
    font-weight: 700 !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #6D28D9 0%,
        #7C3AED 50%,
        #9333EA 100%
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* INPUT BOX */

.stNumberInput input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
}

/* BUTTON */

.stButton>button {

    background: linear-gradient(
        to right,
        #EC4899,
        #F472B6
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 14px !important;

    font-size: 16px !important;

    font-weight: 700 !important;

    padding: 12px 20px !important;

    width: 100% !important;

    transition: 0.3s !important;
}

.stButton>button:hover {

    transform: scale(1.02);

    background: linear-gradient(
        to right,
        #DB2777,
        #EC4899
    ) !important;
}

/* METRIC CARDS */

.metric-card {

    background: white;

    padding: 20px;

    border-radius: 20px;

    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);

    text-align: center;
}

/* CHATBOX */

.chat-box {

    background: linear-gradient(
        135deg,
        #FFF1F7 0%,
        #FFE4F1 100%
    );

    padding: 25px;

    border-radius: 22px;

    border-left: 8px solid #EC4899;

    box-shadow: 0px 6px 18px rgba(236,72,153,0.12);

    color: #1E293B;

    line-height: 1.7;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {

    background: white !important;

    border-radius: 15px !important;
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
