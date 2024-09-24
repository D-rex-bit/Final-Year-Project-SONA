import streamlit as st
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
import joblib

# Load your pre-trained models and vectorizer
rf = joblib.load('random_forest_model.pkl')  # Load your Random Forest model
lo_mo = joblib.load('logistic_regression_model.pkl')  # Load your Logistic Regression model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load your TF-IDF vectorizer

# Download necessary NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess claims
def preprocess_claim(claim):
    claim = re.sub(r'[^a-zA-Z0-9]', ' ', claim)
    claim = ' '.join(lemmatizer.lemmatize(word) for word in claim.lower().split())
    return claim

# Initialize an empty DataFrame to store results in session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Speaker", "Claim", "Random Forest Truth", "Logistic Regression Truth"])

# Streamlit UI
st.title("Claim Truth Detector")

# Input fields for speaker and claim
speaker = st.text_input("Enter Speaker Name:")
claim = st.text_area("Enter Claim:")

if st.button("Submit"):
    if speaker and claim:  # Check if both fields are filled
        # Preprocess the claim
        processed_claim = preprocess_claim(claim)
        
        # Transform the claim using the loaded vectorizer
        claim_vector = tfidf_vectorizer.transform([processed_claim]).toarray()

        # Try to predict using both models
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

            # Create a new DataFrame for the new result
            new_result = pd.DataFrame({
                "Speaker": [speaker],
                "Claim": [claim],
                "Random Forest Truth": [rf_truth_label],
                "Logistic Regression Truth": [lo_mo_truth_label]
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
    st.session_state.results_df = pd.DataFrame(columns=["Speaker", "Claim", "Random Forest Truth", "Logistic Regression Truth"])
    st.experimental_rerun()  # Clear the app state and rerun