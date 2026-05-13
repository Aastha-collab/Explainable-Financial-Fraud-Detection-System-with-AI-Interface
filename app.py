# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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

    # PREDICTION

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

    # TRANSACTION BEHAVIOR

    elif any(word in q for word in [
        "transaction behavior",
        "behaviour",
        "behavior",
        "customer behavior",
        "transaction pattern",
        "spending pattern"
    ]):

        amount = transaction['Amount']

        if prediction == 1:

            return f"""
This transaction shows suspicious behavioral patterns.

The model detected abnormal deviations in features like {', '.join(risk_factors)}.

Transaction amount of {round(amount,2)} and unusual feature activity differ significantly from legitimate customer transaction patterns.
"""

        else:

            return f"""
This transaction shows stable and legitimate customer behavior.

Transaction amount of {round(amount,2)} and feature patterns remain close to normal transaction activity.
"""

    # RISK FACTORS

    elif any(word in q for word in [
        "risk",
        "factor",
        "why"
    ]):

        return f"""
Main suspicious indicators:
{', '.join(risk_factors)}
"""

    # MODEL

    elif any(word in q for word in [
        "model",
        "algorithm"
    ]):

        return """
The system uses Random Forest Machine Learning algorithm for fraud classification.
"""

    # DATASET

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

    # BUSINESS IMPACT

    elif any(word in q for word in [
        "business",
        "impact"
    ]):

        return """
This system helps banks reduce fraud losses,
improve fraud investigation speed,
and enhance explainability in financial systems.
"""

    # DEFAULT

    else:

        return """
I can answer questions related to:

• Fraud prediction
• Risk factors
• Dataset
• Model
• Transaction behavior
• Business impact
"""

# =====================================================
# CHATBOT OUTPUT
# =====================================================

if user_query:

    chatbot_response = smart_chatbot(user_query)

    if user_query:

        chatbot_response = smart_chatbot(user_query)

        st.markdown("### 🤖 AI Fraud Chatbot")

        st.info(chatbot_response)


# =========================================================
# SMALL GRAPHS SECTION
# =========================================================

g1, g2 = st.columns(2)

# =====================================================
# INTERACTIVE VISUALIZATIONS
# =====================================================

st.markdown("## 📊 Interactive Visualizations")

col1, col2 = st.columns(2)

# =====================================================
# BAR GRAPH
# =====================================================

with col2:

    st.markdown("### 📈 Risk Feature Scores")

    selected_features = risk_factors[:5]

    if len(selected_features) == 0:
        selected_features = top_features[:5]

    values = [
        transaction[feature]
        for feature in selected_features
    ]

    bar_df = pd.DataFrame({
        "Feature": selected_features,
        "Value": values
    })

    fig_bar = px.bar(
        bar_df,
        x="Feature",
        y="Value",
        color="Feature",
        color_discrete_sequence=[
            '#EC4899',
            '#A855F7',
            '#6366F1',
            '#06B6D4',
            '#10B981'
        ]
    )

    fig_bar.update_traces(
        hovertemplate=
        "<b>%{x}</b><br>" +
        "Score: %{y}<extra></extra>",
        marker_line_color='white',
        marker_line_width=1.5
    )

    fig_bar.update_layout(
        height=380,
        paper_bgcolor='#0B1120',
        plot_bgcolor='#0B1120',
        font=dict(color='white'),
        xaxis=dict(
            title="Features",
            showgrid=False
        ),
        yaxis=dict(
            title="Feature Score",
            zeroline=True,
            zerolinecolor='white',
            gridcolor='rgba(255,255,255,0.08)'
        ),
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=False
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

# =====================================================
# PIE CHART
# =====================================================

with col1:

    st.markdown("### 🥧 Dataset Distribution")

    pie_df = pd.DataFrame({
        "Category": ["Normal", "Fraud"],
        "Count": [
            len(df[df['Class'] == 0]),
            len(df[df['Class'] == 1])
        ]
    })

    fig_pie = px.pie(
        pie_df,
        names="Category",
        values="Count",
        color="Category",
        color_discrete_map={
            "Normal": "#A855F7",
            "Fraud": "#EC4899"
        },
        hole=0.35
    )

    fig_pie.update_traces(
        textinfo='percent+label',
        hovertemplate=
        "<b>%{label}</b><br>" +
        "Transactions: %{value}<br>" +
        "Percentage: %{percent}",
        marker=dict(
            line=dict(
                color='#FFFFFF',
                width=3
            )
        )
    )

    fig_pie.update_layout(
        height=380,
        paper_bgcolor='#0B1120',
        plot_bgcolor='#0B1120',
        font=dict(color='white'),
        margin=dict(t=30, b=10, l=10, r=10)
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


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
