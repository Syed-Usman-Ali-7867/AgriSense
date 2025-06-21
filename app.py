import streamlit as st
from PIL import Image
import numpy as np
import cv2  # Correct place for importing OpenCV
from utils import predict_disease, recommend_crop

st.set_page_config(page_title="AgriSense AI", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #e8f5e9;
            color: #1b5e20;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
        }
        h1 {
            color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 AgriSense AI")
st.subheader("Intelligent Agriculture Assistant")
st.subheader("Support Smart Farming")

tabs = st.tabs(["🦠 Plant Disease Detection", "🌾 Crop Recommendation"])

# ----- Disease Detection Tab -----
with tabs[0]:
    st.header("Upload a Leaf Image")
    uploaded_file = st.file_uploader("Choose a leaf image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Detecting disease..."):
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            result_img, names, predictions = predict_disease(image_np)

        st.image(result_img, caption="Prediction", use_column_width=True)
        if predictions:
            predicted_labels = [names[int(cls_id)] for cls_id in predictions]
            st.success(f"Disease Detected: **{predicted_labels[0]}**")
        else:
            st.info("No disease detected.")

# ----- Crop Recommendation Tab -----
with tabs[1]:
    st.header("Enter Weather & Soil Info")

    N = st.number_input("Nitrogen (N)", value=80)
    P = st.number_input("Phosphorus (P)", value=7)
    K = st.number_input("Potassium (K)", value=3)
    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=80.0)
    pH = st.number_input("Soil pH", value=6.5)
    rainfall = st.number_input("Rainfall (mm)", value=200.0)

    if st.button("🌱 Recommend Crop"):
        features = [N, P, K, temperature, humidity, pH, rainfall]
        crop = recommend_crop(features)
        st.success(f"✅ Recommended Crop: **{crop.upper()}**")
