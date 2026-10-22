#  Automated Plant Disease Detection Using Custom CNN

A deep learning major project designed to automatically detect and classify plant leaf diseases from images using a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed via an interactive Streamlit web application.

---

##  Project Overview
Plant diseases cause massive agricultural losses worldwide. This project provides a fast, automated AI tool capable of diagnosing 15 different plant conditions from leaf photographs to assist in early detection and treatment.

---

##  Tech Stack & Libraries
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Web Framework:** Streamlit
* **Data Processing:** NumPy, Pandas, Pillow
* **Visualization:** Matplotlib

---

##  Project Structure
```text
Plant-disease-detection/
│
├── Train_cnn.py           # Script for training the custom CNN model
├── predict.py             # CLI script for individual image inference
├── app.py                 # Streamlit web application interface
├── training_performance.png # Training vs Validation accuracy & loss curves
└── README.md              # Project documentation