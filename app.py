import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

# --- 1. Initialize the Flask App (ONLY ONCE) ---
app = Flask(__name__)

# --- 2. Load your trained model ---
# NOTE: Ensure 'stego_detector_model.joblib' is a model that supports predict_proba 
# (like RandomForestClassifier or other ensemble models)
try:
    model = joblib.load('stego_detector_model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 

# --- 3. Define your feature extraction function ---
def extract_features(image_bytes):
    """Reads image bytes and extracts statistical features."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None
        
        # We assume PVD uses the Red channel for steganography
        red_channel = img[:, :, 2]
        
        # Feature 1: Histogram
        hist_values = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
        
        # Feature 2: Difference Histogram (Paired Pixel Difference)
        # Calculates the absolute difference between adjacent pixels
        diff = np.abs(red_channel[:, 1:].astype(np.int16) - red_channel[:, :-1].astype(np.int16)).flatten()
        hist_diff, _ = np.histogram(diff, bins=256, range=(0, 256))
        
        # Concatenate all features into a single vector
        return np.concatenate((hist_values.flatten(), hist_diff.flatten()))
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

# --- 4. Define ALL your page routes ---

# Route for the landing page (Home.html)
@app.route('/')
def home():
    # Placeholder: Assuming the user has a Home.html file
    return render_template('Home.html') 

# Route for the detection page (Detection.html)
@app.route('/detection')
def detection_page():
    return render_template('Detection.html')

# Route for the steganography page (stegnography.html)
@app.route('/steganography')
def steganography_page():
    # Placeholder: Assuming the user has a stegnography.html file
    return render_template('stegnography.html')

# Route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """Receives an image, processes it, and returns a prediction and confidence."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image_bytes = file.read()
        features = extract_features(image_bytes)
        
        if features is not None:
            features = features.reshape(1, -1)
            
            # Predict the class (0 for Clean, 1 for Stego)
            prediction_class = model.predict(features)[0]
            result = 'STEGO' if prediction_class == 1 else 'CLEAN'
            
            # Get prediction probabilities for confidence
            # This is the line that gets the probability for each class
            probabilities = model.predict_proba(features)[0]
            
            # Confidence is the probability of the *predicted* class
            confidence_value = probabilities[prediction_class] * 100
            
            return jsonify({
                'prediction': result,
                'confidence': float(confidence_value)
            })
        else:
            return jsonify({'error': 'Could not process image features'}), 500

# --- 5. Run the App (AT THE VERY END) ---
if __name__ == '__main__':
    # Use reloader=False to prevent model from loading twice
    app.run(debug=True, use_reloader=False)
