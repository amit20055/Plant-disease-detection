import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

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

# 3. Predefined Class Labels
class_labels = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# 4. Web App UI Header
st.title("🌱 Automated Plant Disease Detection")
st.markdown("### College Major Project: Deep Learning using Custom CNN")
st.write("Upload a clear image of a plant leaf below to get an instant diagnosis and confidence score.")

# 5. File Uploader Widget
uploaded_file = st.file_uploader("Choose a leaf image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Leaf Image', use_column_width=True)
    
    # Preprocess Image for Color Validation Check
    img_resized = image_display.resize((128, 128))
    rgb_array = np.array(img_resized)
    
    # Calculate RGB means to check for organic plant color characteristics
    r_mean = np.mean(rgb_array[:, :, 0])
    g_mean = np.mean(rgb_array[:, :, 1])
    b_mean = np.mean(rgb_array[:, :, 2])
    
    # Preprocess for CNN Model Prediction
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    if st.button('Analyze Leaf Health'):
        with st.spinner('Analyzing leaf features through Custom CNN layers...'):
            
            # AUTOMATED GUARD: Check if image lacks natural foliage color balance (e.g., human clothing, skin, indoor objects)
            # Leaves naturally exhibit distinct color ratios compared to random clothing or skin tones.
            if g_mean < 40 or (abs(r_mean - g_mean) < 5 and abs(g_mean - b_mean) < 5):
                st.error("❌ **Invalid Image Detected!**")
                st.warning("The uploaded image does not appear to be a plant leaf. Please upload a clear photograph of a plant leaf.")
            else:
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                detected_disease = class_labels[predicted_class_index]
                
                st.markdown("---")
                st.subheader("📋 Diagnosis Results")
                st.success(f"**Predicted Condition:** `{detected_disease}`")
                st.info(f"**Confidence Score:** `{confidence:.2f}%`")
                
                if "healthy" in detected_disease.lower():
                    st.balloons()
                    st.success("🟢 The leaf appears healthy! No signs of disease detected system-wide.")
                else:
                    st.warning("⚠️ Disease detected! Recommended action: Isolate the affected plant and check agricultural advisory guidelines.")