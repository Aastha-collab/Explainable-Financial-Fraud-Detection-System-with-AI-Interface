# Explainable Financial Fraud Detection System with Risk Analysis and Chatbot Interface

## Project Overview
This project builds an end-to-end financial fraud detection system using machine learning and explainable AI techniques. The system identifies fraudulent transactions and provides human-readable explanations along with an interactive chatbot interface.

The goal is to create an interpretable fraud detection solution suitable for real-world financial applications.

---

## Live Demo
https://explainable-financial-fraud-detection-system-with-ai-interface.streamlit.app

---

## Problem Statement
Financial institutions process millions of transactions daily, making manual fraud detection difficult. Traditional models provide predictions without explanations, reducing trust and interpretability. This project addresses this problem by combining fraud detection with explainable AI and chatbot interaction.

---

## Project Pipeline
Data Loading → Data Cleaning → EDA → Feature Engineering → Train/Test Split → Scaling → Handle Imbalance → Model Training → Model Evaluation → Best Model Selection → Risk Factor Extraction → nomaly Detection (Isolation Forest) →  Explanation Layer → Chatbot Interface

---

## Phase 1: Machine Learning Fraud Detection
- Data preprocessing
- Handling imbalanced dataset
- Logistic Regression model
- Random Forest model
- Model comparison
- Final model selection
- Fraud probability prediction
- Anomaly Detection (Isolation Forest)

Random Forest selected as best model.

- Fraud vs Non- Fraud transaction
<img width="597" height="455" alt="image" src="https://github.com/user-attachments/assets/bbbe93e9-c432-4545-a8ef-57ec64835e9b" />

- Amount Distribution
<img width="566" height="393" alt="image" src="https://github.com/user-attachments/assets/7cae153f-761e-4beb-9805-733e246739c5" />


---

## Phase 2: Explainable AI + Chatbot
- Feature importance extraction
- Risk factor detection
- Explanation generation
- Fraud probability scoring
- Interactive chatbot interface

---

## Example Output

Prediction: Fraud  
Fraud Probability: 0.99  
Risk Factors: V14, V10, V12, V4, V11  
Explanation: Transaction appears suspicious due to unusual patterns in important features.

---

## Features
- Fraud detection using Random Forest
- Explainable AI risk factor analysis
- Prediction confidence scoring
- Interactive chatbot interface
- End-to-end pipeline

---

## Tech Stack
Python  
Scikit-learn  
Pandas  
NumPy  
Matplotlib  
Seaborn  
Streamlit (deployment)  

---
