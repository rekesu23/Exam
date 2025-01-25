import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Upload CSV file
uploaded_file = st.file_uploader("Upload Mushroom Dataset", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data loaded successfully!")

    # Separate features and target (explicitly define target)
    X = df.drop(columns=['class'])
    y = df['class'].map({'e': 1, 'p': 0}) # Map 'e' to 1 and 'p' to 0

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Create and train the model within a pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)


    # Model Evaluation & Saving
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    #Save Model - use os.path.join for better path handling
    model_path = os.path.join(os.getcwd(), 'random_forest_model.pkl')
    joblib.dump(model_pipeline, model_path)
    st.write(f"Model saved to: {model_path}")

    # Load the model (for prediction)
    loaded_model = joblib.load(model_path)

    #Prediction Section
    st.title("Mushroom Classification")
    st.write("Enter the mushroom features:")

    #Get Feature Names after OneHotEncoding
    transformed_X = loaded_model.named_steps['preprocessor'].transform(X)
    feature_names = loaded_model.named_steps['preprocessor'].get_feature_names_out(input_features=X.columns)
    input_values = {}

    for feature in feature_names:
        if feature in loaded_model.named_steps['preprocessor'].transformers_[1][1].categories_[0]:
            input_values[feature] = st.selectbox(feature, loaded_model.named_steps['preprocessor'].transformers_[1][1].categories_[0])
        else:
            input_values[feature] = st.number_input(feature, value=0.0)

    input_df = pd.DataFrame([input_values])
    if st.button('Predict'):
        try:
            prediction = loaded_model.predict(input_df)[0]
            result = "Edible" if prediction == 1 else "Poisonous"
            st.write(f"Prediction: {result}")
        except ValueError as e:
            st.error(f"Error during prediction: {e}")
