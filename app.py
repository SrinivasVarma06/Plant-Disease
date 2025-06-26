import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

model = tf.keras.models.load_model("plant_disease_model.h5")
class_names = [
    "Pepper Bacterial Spot",
    "Pepper Healthy",
    "Potato Late Blight",
    "Tomato Early Blight",
    "Tomato Healthy"
]

st.title("Plant Disease Detection App")
st.write("Upload a leaf image to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure 3 channels
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("Prediction")
    st.write(f"**{predicted_class}**")
    st.write(f"Confidence: {confidence * 100:.2f}%")
