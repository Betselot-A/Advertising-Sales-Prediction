import streamlit as st
import pandas as pd
import joblib

# Load the trained model
artifact = joblib.load("advertising_sales_model.pkl")
model = artifact['model']

# Application title
st.title("Advertising Sales Prediction")
st.write("Predict product sales using TV and Radio advertising budgets.")

# User inputs
tv = st.number_input(
    "TV Advertising Budget (thousands of dollars)",
    min_value=0.0,
    max_value=300.0,
    value=100.0
)

radio = st.number_input(
    "Radio Advertising Budget (thousands of dollars)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

# Prediction button
if st.button("Predict Sales"):
    input_data = pd.DataFrame({
        'TV': [tv],
        'radio': [radio]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Sales: {prediction:.2f} thousand units")