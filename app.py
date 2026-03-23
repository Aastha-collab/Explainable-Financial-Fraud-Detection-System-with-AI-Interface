# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# App title
st.title("Explainable Financial Fraud Detection System")
st.write("This app detects fraudulent transactions and provides explanation with chatbot interaction.")

# Load saved model files
rf_model = joblib.load("rf_model.pkl")
top_features = joblib.load("top_features.pkl")
normal_means = joblib.load("normal_means.pkl")

# Load dataset
df = pd.read_csv("creditcard_sample.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Select transaction
st.subheader("Select Transaction")
index = st.number_input("Transaction Index", 0, len(X)-1, 0)
transaction = X.iloc[index]

# Prediction
prediction = rf_model.predict(transaction.to_frame().T)[0]
proba = rf_model.predict_proba(transaction.to_frame().T)[0][1]

# Risk factor function
def get_risk_factors(transaction, top_features, normal_means, threshold=2):
    risk_factors = []
    
    for feature in top_features:
        value = transaction[feature]
        mean_value = normal_means[feature]
        
        if abs(value - mean_value) > threshold:
            risk_factors.append(feature)
    
    return risk_factors

# Explanation function
def generate_explanation(prediction, risk_factors):
    if prediction == 0:
        if len(risk_factors) == 0:
            return "Transaction appears normal."
        else:
            return f"Transaction is normal but slight deviations in {', '.join(risk_factors)}."
    else:
        if len(risk_factors) == 0:
            return "Transaction flagged suspicious."
        else:
            return f"Transaction suspicious due to {', '.join(risk_factors)}."

# Compute explanation
risk_factors = get_risk_factors(transaction, top_features, normal_means)
explanation = generate_explanation(prediction, risk_factors)

# Display results
st.subheader("Prediction Result")
if prediction == 1:
    st.error("Fraud Transaction")
else:
    st.success("Normal Transaction")

st.write("Fraud Probability:", round(proba,2))

st.write("Risk Factors:", risk_factors)

st.write("Explanation:", explanation)

# Chatbot
st.subheader("Chatbot")
user_query = st.text_input("Ask question")

def chatbot(query):
    q = query.lower()
    
    if "prediction" in q:
        return "Fraud" if prediction == 1 else "Normal"
    
    elif "why" in q:
        return explanation
    
    elif "risk" in q:
        return risk_factors
    
    elif "model" in q:
        return "Random Forest"
    
    else:
        return "Ask about prediction, risk factors, or explanation"

# Chatbot output
if user_query:
    response = chatbot(user_query)
    st.write("Chatbot:", response)
