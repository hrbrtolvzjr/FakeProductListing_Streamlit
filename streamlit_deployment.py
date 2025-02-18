import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load("rfrst_model_final.joblib.pkl")
vectorizer = joblib.load("tf_idf_final.joblib")  # Ensure this file exists

st.title("Fake Listing Detection - Lazada")

# User input
text = st.text_area("Enter product description:")
reviews_total_5_stars = st.number_input("5-star reviews:", min_value=0, step=1)
reviews_total_4_stars = st.number_input("4-star reviews:", min_value=0, step=1)
reviews_total_3_stars = st.number_input("3-star reviews:", min_value=0, step=1)
reviews_total_2_stars = st.number_input("2-star reviews:", min_value=0, step=1)
reviews_total_1_stars = st.number_input("1-star reviews:", min_value=0, step=1)
seller_rating = st.number_input("Seller rating (0-1):", min_value=0.0, max_value=1.0, step=0.01)
ships_on_time = st.number_input("Ships on time (0-1):", min_value=0.0, max_value=1.0, step=0.01)
chat_response_rate = st.number_input("Chat response rate (0-1):", min_value=0.0, max_value=1.0, step=0.01)
price = st.number_input("Price:", min_value=0.0, step=0.01)

if st.button("Predict"):
    if text:
        # Transform input using the loaded vectorizer
        X_text = vectorizer.transform([text])
        X_reviews = np.array([[reviews_total_5_stars, reviews_total_4_stars, reviews_total_3_stars, reviews_total_2_stars, reviews_total_1_stars]])
        X_seller = np.array([[seller_rating, ships_on_time, chat_response_rate]])
        X_combined = hstack((X_text, X_reviews, X_seller))
        
        probability = model.predict_proba(X_combined)
        fake_probability = probability[0, 0]
        
        st.write(f"### Probability of listing being fake: {fake_probability:.2%}")
    else:
        st.warning("Please enter a product description.")
