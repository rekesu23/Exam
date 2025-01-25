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
import joblib #for saving and loading model

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data(df):
    try:
        # Separate features (X) and target (y)
        X = df.drop(columns=['class'])
        y = df['class']

        # Create a column transformer for preprocessing - This is now inside try
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), X.select_dtypes(include=np.number).columns),
                ('cat', OneHotEncoder(), X.select_dtypes(include=['object', 'category']).columns)
            ])
        print(preprocessor)

        # Create a pipeline with preprocessing and model - This is now inside try
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Split data - This is now inside try
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model - This is now inside try
        pipeline.fit(X_train, y_train)

        # Evaluate the model - This is now inside try
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        return pipeline, accuracy, cm, fpr, tpr, roc_auc

    except Exception as e:
        print(f"Error in load_and_preprocess_data: {e}")
        st.error(f"An error occurred during data processing: {e}")
        return None, None, None, None, None, None


# --- Streamlit App ---
st.title("Mushroom Classification App")

uploaded_file = st.file_uploader("Choose a mushrooms.csv file", type="csv")

pipeline = None
df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['class'] = df['class'].map({'e': 0, 'p': 1})
        pipeline, accuracy, cm, fpr, tpr, roc_auc = load_and_preprocess_data(df)

        if pipeline is not None:
            st.subheader("Model Performance")
            st.write(f"Accuracy: {accuracy:.4f}")
            # ... (display results) ...

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")


        # Prediction section
st.subheader("Make a Prediction") 
input_data = {}

if pipeline is not None and df is not None:
    feature_names = df.columns.tolist()
    feature_names.remove('class')

    for feature in feature_names:
        if pd.api.types.is_numeric_dtype(df[feature]):
            input_data[feature] = st.number_input(f"{feature} (numeric)", value=0.0)
        else:
            unique_values = df[feature].unique()
            input_data[feature] = st.selectbox(f"{feature} (categorical)", unique_values)

        if st.button("Predict", key="my_unique_predict_button"):
            input_df = pd.DataFrame([input_data])  # Line 95 - Indented correctly now
            try:
                input_df_processed = pipeline['preprocessor'].transform(input_df)
                prediction = pipeline.predict(input_df_processed)[0]
                st.write(f"Prediction: {prediction}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
else:
    st.warning("Please upload a CSV file to train the model before making a prediction.")
