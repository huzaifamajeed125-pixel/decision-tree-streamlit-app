import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the trained Decision Tree model
model = joblib.load("decision_tree_model.pkl")

st.title("Decision Tree Model Deployment")
st.write(
    "Ye app aapke trained Decision Tree model ka use karke predictions deta hai."
)

# Dataset ke features ke mutabiq input fields
st.subheader("Enter Feature Values:")
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)
feature_4 = st.number_input("Feature 4", value=0.0)

if st.button("Predict"):
  # Input data ko model ke format me array banana
  input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])

  # Prediction karna
  prediction = model.predict(input_data)
  prediction_proba = model.predict_proba(input_data)

  st.success(f"Predicted Class: {prediction[0]}")
  st.write(f"Prediction Probabilities: {prediction_proba}")