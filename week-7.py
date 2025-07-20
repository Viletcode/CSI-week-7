# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
# Example dataset (you can replace this with your own)
data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

X = data.drop("medv", axis=1)
y = data["medv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "house_price_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
# Load model and features
model = joblib.load("house_price_model.pkl")
features = joblib.load("feature_names.pkl")

st.title("🏠 House Price Prediction App")
st.write("Enter the property features below to predict the price.")

# Create input fields dynamically
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame for prediction
input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: ${prediction * 1000:.2f}")

    # Optional: show feature importance (based on coefficients)
    st.subheader("📊 Feature Importance")
    importance = pd.Series(model.coef_, index=features)
    importance = importance.sort_values()

    fig, ax = plt.subplots()
    importance.plot(kind='barh', ax=ax)
    st.pyplot(fig)