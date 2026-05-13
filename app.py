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
    page_title="Explainable Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #071028 0%,
        #111827 40%,
        #1E1B4B 100%
    );
    color: white;
}

/* HIDE STREAMLIT */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* SIDEBAR */

[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #081225 0%,
        #111827 50%,
        #1E1B4B 100%
    );
    border-right:1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] *{
    color:white !important;
}

/* DROPDOWN */

.stSelectbox div[data-baseweb="select"]{
    background:#111827 !important;
    border-radius:14px !important;
    border:2px solid #8B5CF6 !important;
}

/* METRIC CARDS */

.metric-card{
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
    border-radius:22px;
    padding:22px;
    box-shadow:0px 6px 20px rgba(0,0,0,0.35);
    border:1px solid rgba(255,255,255,0.06);
}

/* AI BOX */

.ai-box{
    background: linear-gradient(
        135deg,
        rgba(17,24,39,0.95),
        rgba(30,27,75,0.95)
    );
    padding:28px;
    border-radius:24px;
    border-left:6px solid #22D3EE;
    box-shadow:0px 8px 30px rgba(0,0,0,0.35);
}

/* RESULT BOX */

.result-box{
    background: rgba(255,255,255,0.05);
    border-radius:20px;
    padding:25px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.35);
}

/* BUTTON */

.stButton>button{
    background: linear-gradient(
        to right,
        #EC4899,
        #F472B6
    ) !important;

    color:white !important;
    border:none !important;
    border-radius:14px !important;
    font-weight:700 !important;
    padding:12px 20px !important;

    box-shadow:0px 6px 18px rgba(236,72,153,0.35);
}

.stButton>button:hover{
    transform:scale(1.02);
}

/* CHAT INPUT */

.stTextInput input{
    background:#111827 !important;
    color:white !important;
    border:2px solid #EC4899 !important;
    border-radius:14px !important;
}

/* DATAFRAME */

[data-testid="stDataFrame"]{
    border-radius:15px !important;
    overflow:hidden !important;
}

</style>
""", unsafe_allow_html=True)

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

# FIX HOUR FEATURE ISSUE

if "Hour" not in df.columns:
    df["Hour"] = df["Time"] // 3600

X = df.drop("Class", axis=1)
y = df["Class"]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
# 🔍 Transaction Explorer

### Features

✔ Detect Fraud  
✔ Analyze Risk Factors  
✔ Explain Prediction  
✔ AI Fraud Chatbot  
✔ Interactive Visualizations
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
# RISK FACTORS
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

risk_factors = get_risk_factors(
    transaction,
    top_features,
    normal_means
)

# =========================================================
# EXPLANATION
# =========================================================

def generate_explanation(
    prediction,
    risk_factors
):

    if prediction == 0:

        if len(risk_factors) == 0:

            return """
Transaction behavior appears stable and aligned with normal customer activity.
"""

        else:

            return f"""
Transaction is normal but slight deviations detected in:
{', '.join(risk_factors)}
"""

    else:

        return f"""
Transaction flagged suspicious due to abnormal deviations in:
{', '.join(risk_factors)}
"""

explanation = generate_explanation(
    prediction,
    risk_factors
)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div style="
padding:35px;
border-radius:28px;
background: linear-gradient(
135deg,
#0F172A,
#111827,
#1E1B4B
);
box-shadow:0px 10px 35px rgba(0,0,0,0.45);
margin-bottom:25px;
">

<h1 style="
color:white;
font-size:50px;
font-weight:800;
">
🛡️ Explainable Financial Fraud Detection
</h1>

<p style="
color:#CBD5E1;
font-size:18px;
line-height:1.8;
">

AI-powered fraud detection system that analyzes transaction behavior,
detects suspicious financial patterns,
explains fraud reasoning,
and enables intelligent financial investigation.

</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

cards = [

    ("Total Transactions", len(df)),
    ("Fraud Cases", len(df[df['Class']==1])),
    ("Normal Cases", len(df[df['Class']==0])),
    ("Fraud Probability", f"{round(proba*100,2)}%")
]

for col, (title, value) in zip(
    [m1,m2,m3,m4],
    cards
):

    col.markdown(f"""
    <div class="metric-card">

    <h4 style="color:#CBD5E1;">
    {title}
    </h4>

    <h1 style="
    color:white;
    font-size:38px;
    ">
    {value}
    </h1>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# NEW DASHBOARD LAYOUT
# =========================================================

left, right = st.columns([1,1])

# =========================================================
# LEFT SIDE
# =========================================================

with left:

    st.markdown("## 🎯 Fraud Detection Result")

    st.markdown("""
    <div class="result-box">
    """, unsafe_allow_html=True)

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Normal Transaction")

    st.markdown("### ⚠️ Risk Factors")

    if len(risk_factors) > 0:

        for factor in risk_factors:
            st.markdown(f"- **{factor}**")

    else:

        st.success("No major anomalies detected")

    st.markdown("### 🧠 Fraud Explanation")

    st.write(explanation)

    st.markdown("</div>", unsafe_allow_html=True)

    # TRANSACTION DETAILS

    with st.expander("📄 View Transaction Details"):

        st.dataframe(
            transaction.to_frame().T,
            use_container_width=True
        )

# =========================================================
# RIGHT SIDE
# =========================================================

with right:

    st.markdown("""
    <div class="ai-box">

    <h2 style="
    color:#F9A8D4;
    ">
    🤖 AI Fraud Chatbot
    </h2>

    <p style="
    color:#CBD5E1;
    line-height:1.7;
    ">

    This assistant can answer questions related to:

    • Fraud prediction  
    • Risk factors  
    • Fraud probability  
    • Transaction behavior  
    • Model explanation  
    • Dataset information  
    • Fraud analysis  
    • Business impact  

    </p>

    </div>
    """, unsafe_allow_html=True)

    user_query = st.text_input(
        "Ask anything about the fraud detection project"
    )

    # =====================================================
    # SMART CHATBOT
    # =====================================================

    def smart_chatbot(query):

        q = query.lower()

        if any(word in q for word in [
            "prediction",
            "fraud",
            "result"
        ]):

            if prediction == 1:

                return f"""
This transaction is predicted as FRAUD with probability of {round(proba*100,2)}%.
"""

            else:

                return f"""
This transaction is predicted as NORMAL with low fraud probability of {round(proba*100,2)}%.
"""

        elif any(word in q for word in [
            "risk",
            "factor",
            "why"
        ]):

            return f"""
Main suspicious indicators:
{', '.join(risk_factors)}
"""

        elif any(word in q for word in [
            "model",
            "algorithm"
        ]):

            return """
The system uses Random Forest Machine Learning algorithm for fraud classification.
"""

        elif any(word in q for word in [
            "dataset",
            "data"
        ]):

            return f"""
Dataset contains:

• Total Transactions: {len(df)}
• Fraud Cases: {len(df[df['Class']==1])}
• Normal Cases: {len(df[df['Class']==0])}
"""

        elif any(word in q for word in [
            "business",
            "impact"
        ]):

            return """
This system helps banks reduce fraud losses,
improve fraud investigation speed,
and enhance explainability in financial systems.
"""

        else:

            return """
I can answer questions related to fraud prediction,
risk factors,
dataset,
model,
transaction behavior,
and business impact.
"""

    # =====================================================
    # CHATBOT OUTPUT
    # =====================================================

    if user_query:

        chatbot_response = smart_chatbot(
            user_query
        )

        st.markdown(f"""
<div class="ai-box">

<h3 style="
color:#F472B6;
">
🤖 AI Assistant Response
</h3>

<p style="
color:white;
line-height:1.8;
font-size:17px;
">
{chatbot_response}
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# SMALL GRAPHS SECTION
# =========================================================

g1, g2 = st.columns(2)

# =========================================================
# FEATURE IMPORTANCE GRAPH
# =========================================================

with g1:

    st.markdown("## 📊 Feature Analysis")

    selected_features = ['Amount', 'V14', 'V10']

    values = [
        transaction['Amount'],
        transaction['V14'],
        transaction['V10']
    ]

    fig1, ax1 = plt.subplots(
        figsize=(4,2.5)
    )

    colors = []

    for val in values:

        if val >= 0:
            colors.append("#8B5CF6")
        else:
            colors.append("#EC4899")

    ax1.bar(
        selected_features,
        values,
        color=colors,
        width=0.5
    )

    ax1.axhline(
        y=0,
        color='white',
        linewidth=1
    )

    ax1.set_facecolor("#111827")

    fig1.patch.set_facecolor("#111827")

    ax1.tick_params(
        colors='white',
        labelsize=8
    )

    ax1.set_title(
        "Feature Scores",
        color='white',
        fontsize=10
    )

    st.pyplot(fig1)

# =========================================================
# PIE CHART
# =========================================================

with g2:

    st.markdown("## 📈 Dataset Distribution")

    fig2, ax2 = plt.subplots(
        figsize=(4,2.5)
    )

    ax2.pie(

        [
            len(df[df['Class']==0]),
            len(df[df['Class']==1])
        ],

        labels=['Normal','Fraud'],

        autopct='%1.1f%%',

        colors=['#8B5CF6','#EC4899'],

        textprops={'color':'white','fontsize':8}
    )

    fig2.patch.set_facecolor("#111827")

    st.pyplot(fig2)

# =========================================================
# BUSINESS IMPACT
# =========================================================

st.markdown("""
<div class="ai-box">

<h2 style="
color:#F9A8D4;
">
💼 Business Impact
</h2>

<p style="
color:white;
line-height:1.9;
font-size:17px;
">

• Detect suspicious transactions faster  
• Improve fraud explainability  
• Reduce manual fraud investigation time  
• Enhance AI-driven financial monitoring  
• Strengthen banking security systems  

</p>

</div>
""", unsafe_allow_html=True)
