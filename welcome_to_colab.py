"""# **Data Preprocessing**"""

# Data Preprocessing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
from google.colab import files
uploaded = files.upload()  # Upload `mushrooms.csv` here
df = pd.read_csv('mushrooms.csv')
# Or if you're using Google Colab, you might need to mount your Google Drive first and then specify the path:
# from google.colab import drive
# drive.mount('/content/drive')
# df = pd.read_csv('/content/drive/MyDrive/YourFolder/mushrooms.csv')

# Display the first few rows
df.head()

# Check for missing values
df.isnull().sum()

# Handle missing values (if any), by filling them with the mode or removing them
df.fillna(df.mode().iloc[0], inplace=True)

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df)

# Check the updated dataframe
df.head()



"""# **Featuring Engineering**"""

# Assuming we have numeric features after encoding
scaler = StandardScaler()
# Drop the one-hot encoded class columns instead of the original 'class' column
scaled_features = scaler.fit_transform(df.drop(columns=['class_e', 'class_p']))  # Exclude the target columns

# Combine the scaled features with the target column
df_scaled = pd.DataFrame(scaled_features, columns=df.drop(columns=['class_e', 'class_p']).columns)
df_scaled['class'] = df['class_e'] # Or df['class_p'] depending on your target variable representation

# Example: Polynomial feature creation for numeric data
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(df_scaled.drop(columns=['class']))
df_poly = pd.DataFrame(poly_features)
df_poly['class'] = df_scaled['class']

"""# **Data Analysis and Visualization**"""

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of the target variable 'class'
# Use the one-hot encoded column for plotting
sns.countplot(x='class_e', data=df)  # Replace 'class_e' with 'class_p' if that's your target
plt.title("Class Distribution")
plt.show()

# For the scatter plot, you need to select two numeric features and use the original 'class' column for the hue

# First, find the numeric features (e.g., 'cap-shape_b', 'cap-shape_c' etc.)
# --- The issue was here: select_dtypes was not finding any numeric columns
# --- Solution: Force conversion to numeric after one-hot encoding
numeric_features = df.drop(columns=['class_e', 'class_p']).astype(float).columns
# Select two features from the list of numeric features:
feature1 = numeric_features[0]
feature2 = numeric_features[1]

# Now create the scatter plot with the hue based on the one-hot encoded class column
sns.scatterplot(data=df, x=feature1, y=feature2, hue='class_e') # Replace 'class_e' with 'class_p' if needed
plt.show()



"""# **Model Building**"""

X = df.drop(columns=['class_e', 'class_p'])  # Drop the one-hot encoded class columns
y = df['class_e']  # Or df['class_p'] depending on which represents your target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Initialize classifiers
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

# Train the models
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

"""# **Evaluation and Visualization**"""

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Predict and evaluate models
dt_preds = dt.predict(X_test)
rf_preds = rf.predict(X_test)
lr_preds = lr.predict(X_test)

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Random Forest (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load your trained model (make sure the model is pre-trained and saved)
# You can load a model from a file if you have saved it, for example using joblib or pickle.
# Example: rf = joblib.load('random_forest_model.pkl')
# For simplicity, let's use a random forest classifier here directly.

# Assuming you already have the trained model `rf`
# If you haven't trained it yet, you should train and save it first

# Sample training data
# X_train = <your training data here>
# y_train = <your target data here>
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

# Sample features and model initialization for testing purposes
# Normally, you would load your trained model here

st.title("Mushroom Classification")
st.write("Enter the features of the mushroom to predict whether it's edible or poisonous.")

# Create sample data for feature names (replace with your actual data)
# This should be done based on your actual training data
# Replace `X_train` with the actual data you trained the model on.
feature_names = ['cap-shape', 'cap-color', 'bruises', 'odor']  # Replace with your actual feature names
input_values = {}

# Create input widgets for each feature
for feature in feature_names:
    if feature in ['cap-shape', 'cap-color', 'bruises', 'odor']:  # Example categorical features
        unique_values = ['b', 'e', 's']  # Replace with actual unique values from your training set
        input_values[feature] = st.selectbox(f"{feature} (categorical)", unique_values)
    else:
        input_values[feature] = st.number_input(f"{feature} (numeric)", value=0.0)

# Prediction button
if st.button('Predict'):
    # Create input data for prediction
    input_data = pd.DataFrame([input_values])

    # Feature scaling if necessary (e.g., standardization)
    # If you trained with scaling, do the same here
    # scaler = StandardScaler()
    # input_data_scaled = scaler.transform(input_data)

    # Make prediction using the model (this is just a placeholder for actual model)
    prediction = rf.predict(input_data)[0]

    # Display the prediction
    st.write(f"The prediction is: {prediction}")
