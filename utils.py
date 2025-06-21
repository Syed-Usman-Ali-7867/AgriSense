from ultralytics import YOLO
import cv2
import numpy as np
import joblib

# Load YOLO model once
yolo_model = YOLO('best.pt')

# Load crop recommendation model once
crop_model = joblib.load('crop_recommendation_model.pkl')

def predict_disease(image):
    results = yolo_model(image)
    result_image = results[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    return result_image, results[0].names, results[0].boxes.cls.tolist()

def recommend_crop(input_data):
    prediction = crop_model.predict([input_data])
    return prediction[0]