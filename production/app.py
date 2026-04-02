import cv2
import numpy as np
import joblib
import torch
import logging
import time
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import config
from utils.feature_utils import extract_patch_probabilities, iter_patches_with_coords
from utils.inference_utils import aggregate_patch_decision
from utils.srm_filters import get_srm_filters, apply_srm_filters
from training.train_deep_model import SteganalysisNet

# --- 1. Initialize Flask App ---
app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / "templates"))
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE_BYTES

# --- 1.5. Configure Logging ---
# Fix Unicode encoding issues on Windows by using UTF-8 for console output
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Encode to UTF-8 and write directly to stdout buffer
            sys.stdout.buffer.write((msg + self.terminator).encode('utf-8', errors='replace'))
            sys.stdout.buffer.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stego_detector.log', encoding='utf-8'),
        UTF8StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. Global Model State ---
classical_model = None
classical_scaler = None
deep_model = None
srm_filters = None
device = None
ACTIVE_BACKEND = "classical"  # Global backend state (separate from config)

def load_models():
    """Load models for selected backend."""
    global classical_model, classical_scaler, deep_model, srm_filters, device, ACTIVE_BACKEND
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load classical model (always available as fallback)
    try:
        if config.MODEL_PATH.exists() and config.SCALER_PATH.exists():
            classical_model = joblib.load(config.MODEL_PATH)
            classical_scaler = joblib.load(config.SCALER_PATH)
            logger.info("✓ Classical model loaded (baseline backup)")
        else:
            logger.warning(f"Classical model not found at {config.MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading classical model: {e}", exc_info=True)
    
    # Load deep model (RECOMMENDED - improved from multi-patch training)
    try:
        deep_model_path = config.PROJECT_ROOT / "stego_detector_cnn_model.pt"
        if deep_model_path.exists():
            deep_model = SteganalysisNet(num_filters=30, num_classes=2).to(device)
            deep_model.load_state_dict(torch.load(deep_model_path, map_location=device, weights_only=True), strict=False)
            deep_model.eval()
            srm_filters = get_srm_filters()
            ACTIVE_BACKEND = "deep_cnn"  # SET GLOBAL BACKEND
            logger.info(f"✓ Deep CNN model loaded on {device}")
            logger.info(f"  Model: CNN with SRM preprocessing")
            logger.info(f"  Aggregation: Multi-patch averaging")
            logger.info(f"  Threshold: {config.MEAN_PROB_THRESHOLD}")
            logger.info(f"  Performance: 39% FPR, 122/200 clean images correctly identified")
        else:
            logger.warning(f"Deep model not found at {deep_model_path}")
            logger.info("Falling back to classical model")
            ACTIVE_BACKEND = "classical"
    except Exception as e:
        logger.error(f"Error loading deep model: {e}", exc_info=True)
        logger.info("Falling back to classical model")
        ACTIVE_BACKEND = "classical"

# Load on startup
print("=" * 60)
print("STEGO DETECTOR - Initializing models")
print("=" * 60)
load_models()
print(f"Active backend: {ACTIVE_BACKEND}")
print("=" * 60)

# --- 4. Define ALL your page routes ---

# Health check endpoint for monitoring
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    model_status = {
        'deep_model_loaded': deep_model is not None,
        'classical_model_loaded': classical_model is not None,
        'backend': ACTIVE_BACKEND
    }
    
    # Determine overall health
    if ACTIVE_BACKEND == "deep_cnn" and deep_model is not None:
        status = 'healthy'
    elif ACTIVE_BACKEND == "classical" and classical_model is not None:
        status = 'healthy'
    else:
        status = 'degraded'
    
    response = {
        'status': status,
        'model_backend': ACTIVE_BACKEND,
        'models': model_status,
        'device': device,
        'threshold': config.MEAN_PROB_THRESHOLD,
        'aggregation_policy': config.AGGREGATION_POLICY
    }
    
    return jsonify(response), 200 if status == 'healthy' else 503

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
    """Receives an image and returns prediction using active backend."""
    with open("debug.log", "a") as f:
        f.write(f"\n=== PREDICT CALLED ===\n")
        f.write(f"Backend: {ACTIVE_BACKEND}\n")
        f.write(f"deep_model is None: {deep_model is None}\n")
        f.write(f"classical_model is None: {classical_model is None}\n")
    
    start_time = time.time()
    
    # Check file upload
    if 'file' not in request.files:
        logger.warning("Prediction request missing file")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("Prediction request with empty filename")
        return jsonify({'error': 'No selected file'}), 400

    logger.info(f"Processing file: {file.filename}")
    
    # Read image
    image_bytes = file.read()
    if not image_bytes:
        logger.warning(f"Empty upload for file: {file.filename}")
        return jsonify({'error': 'Empty upload'}), 400

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        logger.error(f"Failed to decode image: {file.filename}")
        return jsonify({'error': 'Failed to decode image'}), 400
    
    logger.info(f"Image decoded: {img.shape[1]}x{img.shape[0]}px")
    
    with open("debug.log", "a") as f:
        f.write(f"Image shape: {img.shape}\n")
    
    # Use appropriate backend
    if ACTIVE_BACKEND == "deep_cnn" and deep_model is not None:
        with open("debug.log", "a") as f:
            f.write(f"Calling predict_deep\n")
        result = predict_deep(img)
    else:
        with open("debug.log", "a") as f:
            f.write(f"Calling predict_classical (backend={ACTIVE_BACKEND})\n")
        result = predict_classical(img)
    
    with open("debug.log", "a") as f:
        f.write(f"Result: {result}\n")
    
    if result is None:
        logger.error(f"Prediction failed for file: {file.filename}")
        return jsonify({'error': 'Prediction failed'}), 500
    
    # Log prediction result
    elapsed_time = time.time() - start_time
    logger.info(f"Prediction: {result['prediction']}, Confidence: {result.get('confidence', 0):.2f}%, Time: {elapsed_time:.3f}s")
    
    # Add timing to result
    result['processing_time_seconds'] = round(elapsed_time, 3)
    
    return jsonify(result)


def predict_classical(img):
    """Predict using classical RandomForest model."""
    if classical_model is None or classical_scaler is None:
        return None
    
    try:
        # Preprocess image
        from utils.feature_utils import preprocess_image
        img, was_resized = preprocess_image(img, max_dimension=2048)
        
        if was_resized:
            logger.info(f"Image resized for classical model processing")
        
        patch_probs = extract_patch_probabilities(img, model=classical_model, scaler=classical_scaler)
        summary = aggregate_patch_decision(patch_probs)
        
        return {
            'prediction': summary['result'],
            'confidence': summary['confidence'],
            'backend': 'classical',
            'max_patch_probability': summary['max_prob'],
            'mean_patch_probability': summary['mean_prob'],
            'suspicious_patch_ratio': summary['suspicious_ratio'],
            'suspicious_patches': summary['suspicious_patches'],
            'total_patches': summary['total_patches']
        }
    except Exception as e:
        logger.error(f"Error in classical prediction: {e}", exc_info=True)
        return None


def predict_deep(img):
    """Predict using deep CNN model."""
    debug_log = []
    debug_log.append(f"predict_deep called")
    
    if deep_model is None or srm_filters is None:
        debug_log.append(f"Models not initialized")
        with open("debug.log", "a") as f:
            f.write("\n".join(debug_log) + "\n")
        logger.error("Deep model or SRM filters not initialized")
        return None
    
    try:
        # Preprocess image (resize if too large, handle edge cases)
        from utils.feature_utils import preprocess_image
        debug_log.append(f"Input image shape: {img.shape}")
        
        img, was_resized = preprocess_image(img, max_dimension=2048)
        debug_log.append(f"After preprocess shape: {img.shape}, resized={was_resized}")
        
        if was_resized:
            logger.info(f"Image resized to {img.shape[1]}x{img.shape[0]} for processing")
        
        # Validate image shape
        if img.shape[2] != 3:
            debug_log.append(f"ERROR: Image has {img.shape[2]} channels")
            logger.error(f"Image has {img.shape[2]} channels, expected 3")
            return None
        
        # Extract patches
        patches = []
        H, W = img.shape[:2]
        patch_size = config.PATCH_SIZE
        
        if H < patch_size or W < patch_size:
            logger.warning(f"Image too small ({H}x{W}), using as single patch")
            patches.append(img)
        else:
            for y in range(0, H - patch_size + 1, patch_size):
                for x in range(0, W - patch_size + 1, patch_size):
                    patch = img[y:y+patch_size, x:x+patch_size]
                    if patch.shape == (patch_size, patch_size, 3):
                        patches.append(patch)
        
        debug_log.append(f"Extracted {len(patches)} patches")
        
        if not patches:
            debug_log.append(f"No valid patches")
            logger.error("No valid patches extracted from image")
            return None
        
        logger.info(f"Extracted {len(patches)} patches for prediction")
        
        # Process ALL patches and average predictions (consistent with training)
        all_stego_probs = []
        
        for patch_idx, patch in enumerate(patches):
            try:
                residuals = apply_srm_filters(patch, srm_filters)
                residuals_tensor = torch.from_numpy(residuals.transpose(2, 0, 1)).float().unsqueeze(0)
                
                # Move tensor to same device as model
                residuals_tensor = residuals_tensor.to(device)
                
                with torch.no_grad():
                    logits = deep_model(residuals_tensor)
                    probs = torch.softmax(logits, dim=1)
                    stego_prob = probs[0, 1].item()
                    all_stego_probs.append(stego_prob)
            except Exception as e:
                debug_log.append(f"Error in patch {patch_idx}: {str(e)}")
                logger.error(f"Error processing patch {patch_idx}: {e}", exc_info=True)
                raise
        
        debug_log.append(f"Computed {len(all_stego_probs)} probabilities")
        
        if not all_stego_probs:
            debug_log.append(f"No stego probs")
            logger.error("No stego probabilities computed")
            return None
        
        # Apply MEAN aggregation policy (as per Experiment 2)
        mean_stego_prob = np.mean(all_stego_probs)
        
        # Apply threshold (as per Experiment 3)
        prediction = "STEGO" if mean_stego_prob > config.MEAN_PROB_THRESHOLD else "CLEAN"
        
        debug_log.append(f"SUCCESS: mean_prob={mean_stego_prob:.4f}, prediction={prediction}")
        
        logger.info(f"Prediction computed: mean_prob={mean_stego_prob:.4f}, prediction={prediction}")
        
        return {
            'prediction': prediction,
            'confidence': max(mean_stego_prob, 1 - mean_stego_prob) * 100,
            'backend': 'deep_cnn',
            'patch_count': len(patches),
            'mean_stego_probability': float(mean_stego_prob) * 100,
            'stego_probability': float(mean_stego_prob) * 100,
            'clean_probability': float(1 - mean_stego_prob) * 100,
            'aggregation_policy': 'mean',
            'threshold': config.MEAN_PROB_THRESHOLD
        }
    except Exception as e:
        debug_log.append(f"FATAL: {str(e)}")
        logger.error(f"Error in deep prediction: {e}", exc_info=True)
        return None
    finally:
        # Write debug log
        with open("debug.log", "a") as f:
            f.write("\n".join(debug_log) + "\n\n")

# --- 5. Run the App (AT THE VERY END) ---
if __name__ == '__main__':
    # Use reloader=False to prevent model from loading twice
    app.run(debug=False, use_reloader=False)
