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
        # ... (your preprocessing code) ...
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        return pipeline, accuracy, cm, fpr, tpr, roc_auc
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {e}")
        return None, None, None, None, None, None #Return None for all values if error

# --- Streamlit App ---
st.title("Mushroom Classification App")

uploaded_file = st.file_uploader("Choose a mushrooms.csv file", type="csv")

pipeline = None  # Initialize pipeline outside the if block

if uploaded_file is not None:
    try:
        pipeline, accuracy, cm, fpr, tpr, roc_auc = load_and_preprocess_data(uploaded_file)

    if pipeline is None:  #Check for errors in preprocessing
        st.error("An error occurred during data processing.  See console for details.")
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
