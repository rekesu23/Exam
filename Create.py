import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("DataFrame shape:", df.shape)
        print("Column names:", df.columns)
        X = df.drop(columns=['class'])
        y = df['class'].map({'e': 0, 'p': 1})  #Convert to numerical
        return X, y #Return X and y directly
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# --- Streamlit App ---
st.title("Mushroom Classification App")

uploaded_file = st.file_uploader("Choose a mushrooms.csv file", type="csv")

pipeline = None  # Initialize pipeline outside the if block

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Error: Uploaded CSV file is empty.")
        else:
            # ... (preprocessing and model loading) ...
            pipeline, accuracy, cm, fpr, tpr, roc_auc = load_and_preprocess_data(uploaded_file)
            # ... (displaying results) ...

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")

#Prediction section
st.subheader("Make a Prediction")
input_data = {} # Define input_data outside the if block

if pipeline is not None:
    feature_names = df.columns.tolist()
    feature_names.remove('class')

    for feature in feature_names:
        if pd.api.types.is_numeric_dtype(df[feature]):
            input_data[feature] = st.number_input(f"{feature} (numeric)", value=0.0)
        else:
            unique_values = df[feature].unique()
            input_data[feature] = st.selectbox(f"{feature} (categorical)", unique_values)


    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])  #Now input_data is defined
        try:
            prediction = pipeline.predict(input_df)[0]
            st.write(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
else:
    st.warning("Please upload a CSV file to train the model before making a prediction.")
