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
    df = pd.read_csv(filepath)

    #Convert target variable to numerical:  Crucial change here!
    df['class'] = df['class'].map({'e': 0, 'p': 1}) #'e'=edible, 'p'=poisonous

    X = df.drop(columns=['class'])
    y = df['class']

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=np.number).columns),
            ('cat', OneHotEncoder(), X.select_dtypes(include=['object', 'category']).columns)
        ])

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42)) #You can change classifier here.
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Train the model
    pipeline.fit(X_train, y_train)

    #Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    return pipeline, accuracy, cm, fpr, tpr, roc_auc


# --- Streamlit App ---
st.title("Mushroom Classification App")

uploaded_file = st.file_uploader("Choose a mushrooms.csv file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)  # Read the CSV directly

        #Check for empty file
        if df.empty:
            st.error("Error: Uploaded CSV file is empty.")
        else:
            # Dynamically get feature names.  This is crucial!
            feature_names = df.columns.tolist()
            feature_names.remove('class') #Remove target column

            # ... (rest of your preprocessing and model loading) ...

            st.subheader("Make a Prediction")
            input_data = {}
            for feature in feature_names:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    input_data[feature] = st.number_input(f"{feature} (numeric)", value=0.0)
                else:  #Handle categorical features
                    unique_values = df[feature].unique()
                    input_data[feature] = st.selectbox(f"{feature} (categorical)", unique_values)

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                try:
                    prediction = pipeline.predict(input_df)[0]
                    st.write(f"Prediction: {prediction}")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    except pd.errors.EmptyDataError:
        st.error("Error: Uploaded CSV file is empty or corrupted.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Awaiting for mushrooms.csv file to be uploaded.")
