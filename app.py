import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os

# Load model (ensure you use raw string for Windows path)
model = load_model(r'C:\Fruits-Image\Image_classify.keras')

# Category labels
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

img_height = 180
img_width = 180

st.header('Fruit & Vegetable Image Classification')
image_name = st.text_input('Enter image file name (e.g., corn.jpg):', 'corn.jpg')

# You can either hardcode or let user upload (optional)
image_path = os.path.join(os.getcwd(), image_name)

if os.path.exists(image_path):
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)  # Proper way to convert to array
    img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

    prediction = model.predict(img_bat)
    predicted_label = data_cat[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.image(image_path, width=200)
    st.write(f" Prediction: **{predicted_label}**")
    st.write(f" Confidence: **{confidence:.2f}%**")
else:
    st.error("❌ Image file not found. Make sure it's in the working directory.")
