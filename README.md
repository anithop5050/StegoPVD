🕵️ Steganography Detection Using PVD

📌 Overview

This mini project focuses on detecting hidden information in digital images using a machine learning–based steganalysis approach. The system is built around the Pixel Value Differencing (PVD) technique, where statistical features are extracted from pixel intensity differences (especially from the red color channel) to identify patterns caused by data embedding.

A Flask-based web application is developed to provide a user-friendly interface where users can upload images. The uploaded image is processed, features are extracted, and a trained machine learning model predicts whether the image is CLEAN (no hidden data) or STEGO (contains hidden data) along with a confidence score.

🛠️ Technologies

Python,
Flask,
OpenCV,
NumPy,
Scikit-learn,
Joblib,
HTML / CSS

⚙️ How It Works

User uploads an image

Red-channel features and PVD differences are extracted

Trained ML model predicts:

CLEAN or STEGO

Confidence percentage

🚀 Run the Project

git clone https://github.com/anithop5050/StegoPVD.git

cd StegoPVD

pip install flask numpy opencv-python scikit-learn joblib

python app.py

🎓 Project Info

Type: Mini Project

Domain: Image Processing / Cyber Security

Use: Academic only
