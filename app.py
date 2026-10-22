import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# 2. Load the Trained Model (Cached for performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_cnn_model.h5')

with st.spinner("Loading Custom CNN Model... Please wait."):
    model = load_model()

# 3. Automatically Load Class Labels from PlantVillage Directory
data_dir = 'PlantVillage'
class_labels = sorted(os.listdir(data_dir))

# 4. Web App UI Header
st.title("🌱 Automated Plant Disease Detection")
st.markdown("### College Major Project: Deep Learning using Custom CNN")
st.write("Upload a clear image of a plant leaf below to get an instant diagnosis and confidence score.")

# 5. File Uploader Widget
uploaded_file = st.file_uploader("Choose a leaf image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Leaf Image', use_column_width=True)
    
    # Preprocess Image for the Custom CNN Model
    img = image_display.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Convert to batch of 1
    img_array /= 255.0 # Rescale pixel values
    
    # Predict Button
    if st.button('Analyze Leaf Health'):
        with st.spinner('Analyzing leaf features through Custom CNN layers...'):
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            detected_disease = class_labels[predicted_class_index]
            
            # Display Results Cleanly
            st.markdown("---")
            st.subheader("📋 Diagnosis Results")
            st.success(f"**Predicted Condition:** `{detected_disease}`")
            st.info(f"**Confidence Score:** `{confidence:.2f}%`")
            
            # Condition-based response feedback
            if "healthy" in detected_disease.lower():
                st.balloons()
                st.success("🟢 The leaf appears healthy! No signs of disease detected.")
            else:
                st.warning("⚠️ Disease detected! Recommended action: Isolate the affected plant and check agricultural advisory guidelines.")