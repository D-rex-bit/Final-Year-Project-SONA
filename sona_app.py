import streamlit as st
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import joblib

# Load your pre-trained models and vectorizer
rf = joblib.load('random_forest_model.pkl')
lo_mo = joblib.load('logistic_regression_model.pkl') 
gb_model = joblib.load('gradient_boosting_model.pkl')  
dt_model = joblib.load('decision_tree_model.pkl')  
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  



# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess claims
def preprocess_claim(claim):
    claim = re.sub(r'[^a-zA-Z0-9]', ' ', claim)  # Remove special characters
    claim = ' '.join(lemmatizer.lemmatize(word) for word in claim.lower().split())  # Lemmatize and convert to lower case
    return claim

# Initialize an empty DataFrame to store results in session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Speaker", "Claim", "Random Forest Truth", "Logistic Regression Truth", "Gradient Boosting Truth", "Decision Tree Truth"])

# Streamlit UI
st.title("S.O.N.A Analysis")

# Input fields for speaker and claim
speaker = st.text_input("Enter Speaker Name:")
claim = st.text_area("Enter Claim:")

if st.button("Submit"):
    if speaker and claim:  # Check if both fields are filled
        # Preprocess the claim
        processed_claim = preprocess_claim(claim)
        
        # Transform the claim using the loaded vectorizer
        claim_vector = tfidf_vectorizer.transform([processed_claim]).toarray()

        # Try to predict using all models
        try:
            # Assuming encoded speaker is mapped correctly (0 for simplicity)
            encoded_speaker_value = np.array([[0]])  # Replace with actual logic for encoding speaker
            X_combined = np.hstack([claim_vector, encoded_speaker_value])  # Combine features

            # Predict with Random Forest
            rf_truth_value = rf.predict(X_combined)
            rf_truth_label = "True" if rf_truth_value[0] == 1 else "False"

            # Predict with Logistic Regression
            lo_mo_truth_value = lo_mo.predict(X_combined)
            lo_mo_truth_label = "True" if lo_mo_truth_value[0] == 1 else "False"

            # Predict with Gradient Boosting
            gb_truth_value = gb_model.predict(X_combined)
            gb_truth_label = "True" if gb_truth_value[0] == 1 else "False"

            # Predict with Decision Tree
            dt_truth_value = dt_model.predict(X_combined)
            dt_truth_label = "True" if dt_truth_value[0] == 1 else "False"

            # Create a new DataFrame for the new result
            new_result = pd.DataFrame({
                "Speaker": [speaker],
                "Claim": [claim],
                "Random Forest Truth": [rf_truth_label],
                "Logistic Regression Truth": [lo_mo_truth_label],
                "Gradient Boosting Truth": [gb_truth_label],
                "Decision Tree Truth": [dt_truth_label]
            })

            # Concatenate the new result with the existing DataFrame in session state
            st.session_state.results_df = pd.concat([st.session_state.results_df, new_result], ignore_index=True)

            # Display results in a table
            st.write("### Results")
            st.table(st.session_state.results_df)

        except ValueError as e:
            # Handle ValueError related to feature mismatch
            st.error(f"Error: {str(e)}. Please check if the input data matches the model's expected features.")

    else:
        st.warning("Please enter both speaker and claim.")  # Input validation warning

# Optionally, you can add a way to clear the input fields and results
if st.button("Clear"):
    st.session_state.results_df = pd.DataFrame()