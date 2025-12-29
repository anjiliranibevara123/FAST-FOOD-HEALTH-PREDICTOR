# Fast Food Health Impact Predictor

A Streamlit web application that predicts a person's overall health score based on their fast food consumption habits and lifestyle data using machine learning.

## Features

- Interactive web interface for health prediction
- Machine learning model trained on lifestyle and dietary data
- Real-time predictions based on user inputs
- Clean, user-friendly Streamlit interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fast-food-health-predictor.git
cd fast-food-health-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open the Streamlit app in your browser
2. Enter your personal details (age, gender, BMI, etc.)
3. Input your fast food consumption and lifestyle habits
4. Click "Predict Health Score" to get your prediction

## Model Details

The prediction model uses:
- Random Forest Regressor
- Features: Age, Gender, Fast Food Meals Per Week, Daily Calories, BMI, Physical Activity, Sleep Hours, Energy Level, Digestive Issues, Doctor Visits
- Preprocessing: Standard scaling and feature engineering

## Deployment

This app can be deployed on:
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Any platform supporting Python/Streamlit

## Dataset

The model was trained on a comprehensive dataset of fast food consumption and health impact data.

## License

MIT License