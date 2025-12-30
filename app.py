import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="Fast Food Health Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background and colors
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
    }
    .stApp {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        border: 2px solid #ffa500;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff6b6b, #ffa500, #ffd700);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background: rgba(255, 255, 255, 0.9);
        color: #333;
        border: 2px solid #ffa500;
        border-radius: 8px;
        font-size: 14px;
    }
    h1, h2, h3 {
        color: #ff4500;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .prediction-highlight {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        animation: pulse 2s infinite, glow 2s infinite alternate;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(0, 0, 0, 0.3); }
        to { box-shadow: 0 0 30px rgba(255, 165, 0, 0.8); }
    }
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load the model and scaler
model = joblib.load("health_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title
st.title("🍔 Fast Food Health Impact Predictor")
st.write("🌟 Predict your overall health score based on lifestyle and fast food consumption. Discover insights to improve your well-being!")

# Add images
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="🍟 Fast Food Culture", width=300)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="🥗 Healthy Lifestyle", width=300)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="🏃‍♂️ Active Living", width=300)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for additional info
st.sidebar.header("ℹ️ About")
st.sidebar.write("🩺 This app predicts your health score based on your habits.")
st.sidebar.write("📊 Enter your details below to get a personalized prediction.")
st.sidebar.image("https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80", caption="🏥 Health Awareness", width=250)
st.sidebar.markdown("---")
st.sidebar.write("🔥 **Fun Fact:** Fast food can impact your health more than you think!")
st.sidebar.write("💪 Take control of your wellness today!")

# Input fields
st.header("📝 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    fast_food = st.number_input("🍔 Fast Food Meals Per Week", min_value=0, value=5)
    calories = st.number_input("🔥 Average Daily Calories", min_value=0, value=2000)

with col2:
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0)
    activity = st.number_input("🏃‍♂️ Physical Activity Hours Per Week", min_value=0.0, value=5.0)
    sleep = st.number_input("😴 Sleep Hours Per Day", min_value=0.0, max_value=24.0, value=8.0)
    energy = st.number_input("⚡ Energy Level Score (1-10)", min_value=1, max_value=10, value=5)

digestive = st.selectbox("🤢 Digestive Issues", ["Yes", "No"])
doctor = st.number_input("👨‍⚕️ Doctor Visits Per Year", min_value=0, value=2)

# When predict button
if st.button("🔮 Predict Health Score"):
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
    
    # Display result with color and highlight based on score
    if prediction >= 7:
        st.markdown(f'<div class="prediction-highlight" style="background: linear-gradient(45deg, #4CAF50, #8BC34A); color: white;">🏆 Excellent Health! Predicted Overall Health Score: {prediction:.2f}</div>', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="🎉 Your Health is Excellent! Keep it up!", width=250)
        st.balloons()
    elif prediction >= 5:
        st.markdown(f'<div class="prediction-highlight" style="background: linear-gradient(45deg, #2196F3, #00BCD4); color: white;">👍 Good Health. Predicted Overall Health Score: {prediction:.2f}</div>', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="👍 Your Health is Good!", width=250)
    elif prediction >= 3:
        st.markdown(f'<div class="prediction-highlight" style="background: linear-gradient(45deg, #FF9800, #FFC107); color: black;">⚠️ Moderate Health. Predicted Overall Health Score: {prediction:.2f}</div>', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="⚠️ Your Health Needs Attention", width=250)
    else:
        st.markdown(f'<div class="prediction-highlight" style="background: linear-gradient(45deg, #F44336, #E91E63); color: white;">🚨 Poor Health. Predicted Overall Health Score: {prediction:.2f}</div>', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="🚨 Take Care of Your Health!", width=250)
    
    st.write("💡 Note: Higher scores indicate better overall health.")
    
    # Health Tips Section
    with st.expander("🌱 Health Improvement Tips"):
        st.write("**Based on your prediction, here are some personalized tips:**")
        if prediction >= 7:
            st.write("🎉 Keep up the great work! Maintain your healthy habits.")
        elif prediction >= 5:
            st.write("👍 Consider reducing fast food intake and increasing physical activity.")
        elif prediction >= 3:
            st.write("⚠️ Focus on balanced meals, regular exercise, and adequate sleep.")
        else:
            st.write("🚨 **Poor Health Alert!** Your current habits may be negatively impacting your health.")
            st.header("⚠️ Potential Health Effects:")
            st.write("• Increased risk of obesity and related conditions")
            st.write("• Higher chances of heart disease and diabetes")
            st.write("• Digestive problems and nutrient deficiencies")
            st.write("• Reduced energy levels and mental health issues")
            st.write("• Weakened immune system")
            st.header("🛡️ Recommended Precautions:")
            st.write("• **Diet Changes:** Reduce fast food to 1-2 times per week, focus on whole foods, fruits, vegetables, and lean proteins")
            st.write("• **Exercise:** Aim for 150 minutes of moderate activity per week (walking, cycling, swimming)")
            st.write("• **Sleep:** Get 7-9 hours of quality sleep nightly")
            st.write("• **Medical Check-ups:** Schedule regular doctor visits for monitoring")
            st.write("• **Hydration:** Drink at least 8 glasses of water daily")
            st.write("• **Stress Management:** Practice mindfulness or hobbies to reduce stress")
        st.write("🍎 Remember: Small changes can lead to big improvements in your health!")