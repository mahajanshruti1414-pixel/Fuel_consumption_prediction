import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Fuel Consumption Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 AI Fuel Consumption Predictor")
st.write("Input vehicle specification parameters to forecast fuel efficiency.")

# Load serialized model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('fuel_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.success("Trained model and scaler loaded successfully!")
except Exception as e:
    st.error("Error loading model files. Make sure 'fuel_model.joblib' and 'scaler.joblib' are in the root directory.")
    st.stop()

st.subheader("Vehicle Specifications")

col1, col2 = st.columns(2)

with col1:
    cylinders = st.selectbox("Number of Cylinders", options=[4, 6, 8], index=0)
    displacement = st.slider("Engine Displacement (cu. in.)", min_value=100.0, max_value=450.0, value=250.0)
    horsepower = st.slider("Horsepower (HP)", min_value=70.0, max_value=230.0, value=130.0)

with col2:
    weight = st.slider("Vehicle Weight (lbs)", min_value=1800.0, max_value=5000.0, value=3000.0)
    acceleration = st.slider("0-60 mph Acceleration (sec)", min_value=8.0, max_value=25.0, value=15.0)

# Prediction Logic
if st.button("Predict Fuel Consumption", type="primary"):
    # Create input dataframe matching feature names
    input_data = pd.DataFrame({
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration]
    })
    
    # Scale inputs using saved scaler
    scaled_input = scaler.transform(input_data)
    
    # Predict
    predicted_consumption = model.predict(scaled_input)[0]
    
    st.divider()
    st.metric(
        label="Estimated Fuel Consumption", 
        value=f"{predicted_consumption:.2f} L/100km"
    )
    
    # Contextual feedback
    if predicted_consumption < 8.0:
        st.info("💡 Highly Efficient Vehicle")
    elif predicted_consumption < 13.0:
        st.warning("⚠️ Average Fuel Efficiency")
    else:
        st.error("🚨 High Fuel Consumption Engine")