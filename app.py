import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("health_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title
st.title("Fast Food Health Impact Predictor")
st.write("Predict your overall health score based on lifestyle and fast food consumption.")

# Sidebar for additional info
st.sidebar.header("About")
st.sidebar.write("This app predicts your health score based on your habits.")
st.sidebar.write("Enter your details below to get a prediction.")

# Input fields
st.header("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    fast_food = st.number_input("Fast Food Meals Per Week", min_value=0, value=5)
    calories = st.number_input("Average Daily Calories", min_value=0, value=2000)

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    activity = st.number_input("Physical Activity Hours Per Week", min_value=0.0, value=5.0)
    sleep = st.number_input("Sleep Hours Per Day", min_value=0.0, max_value=24.0, value=8.0)
    energy = st.number_input("Energy Level Score (1-10)", min_value=1, max_value=10, value=5)

digestive = st.selectbox("Digestive Issues", ["Yes", "No"])
doctor = st.number_input("Doctor Visits Per Year", min_value=0, value=2)

# When predict button
if st.button("Predict Health Score"):
    # Create input df
    input_data = {
        'Age': age,
        'Gender': gender,
        'Fast_Food_Meals_Per_Week': fast_food,
        'Average_Daily_Calories': calories,
        'BMI': bmi,
        'Physical_Activity_Hours_Per_Week': activity,
        'Sleep_Hours_Per_Day': sleep,
        'Energy_Level_Score': energy,
        'Digestive_Issues': digestive,
        'Doctor_Visits_Per_Year': doctor
    }
    df_input = pd.DataFrame([input_data])
    
    # Preprocessing
    # Get dummies
    df_input = pd.get_dummies(df_input, drop_first=True)
    
    # Feature engineering
    df_input['Lifestyle_Index'] = df_input['Fast_Food_Meals_Per_Week'] / (df_input['Physical_Activity_Hours_Per_Week'] + 1)
    
    # Ensure same columns as training
    df_input = df_input.reindex(columns=feature_names, fill_value=0)
    
    # Scale
    X_input_scaled = scaler.transform(df_input)
    
    # Predict
    prediction = model.predict(X_input_scaled)[0]
    
    st.write(f"Predicted Overall Health Score: {prediction:.2f}")
    st.write("Note: Higher scores indicate better overall health.")