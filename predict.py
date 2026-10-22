import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 1. Load your newly trained model
print("[INFO] Loading trained model...")
model = tf.keras.models.load_model('plant_cnn_model.h5')



# 2. Automatically get class labels from your PlantVillage folder
data_dir = 'PlantVillage'
class_labels = sorted(os.listdir(data_dir))
print(f"[INFO] Loaded {len(class_labels)} classes successfully.")

# 3. Path to the test leaf image you want to check
# (Replace 'test_leaf.jpg' with the actual path or filename of any leaf image on your computer)
img_path = r'C:\Users\amitp\OneDrive\Desktop\Plant_Disease_Project\PlantVillage\Tomato_Spider_mites_Two_spotted_spider_mite\0a2b122a-0f23-4c92-8b7a-21b274dffbe5___Com.G_SpM_FL 9350.JPG'

if not os.path.exists(img_path):
    print(f"\n[ERROR] Could not find the image at '{img_path}'. Please place a test image in your project folder and update the 'img_path' variable!")
else:
    # 4. Preprocess the image to match what the model expects (128x128 pixels, rescaled)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Convert single image to a batch of 1
    img_array /= 255.0 # Rescale pixel values between 0 and 1

    # 5. Make the prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    # 6. Output the results
    print("\n==============================")
    print("      DIAGNOSIS RESULTS       ")
    print("==============================")
    print(f"Predicted Disease: {class_labels[predicted_class_index]}")
    print(f"Confidence Score:  {confidence:.2f}%")
    print("==============================\n")