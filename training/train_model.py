import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import sys
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import CLEAN_TRAIN_DIRS, IMAGE_EXTENSIONS, MODEL_PATH, SCALER_PATH, STEGO_TRAIN_DIRS
from utils.feature_utils import extract_patch_features

def list_image_files(directory: Path):
    if not directory.exists():
        return []
    return [
        file_path
        for file_path in directory.glob("*")
        if file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]

def extract_training_features(img_path, is_clean):
    """Extracts features from multiple patches if clean, or top-left patch if stego."""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None: 
            return []
            
        H, W = img.shape[:2]
        if H < 128 or W < 128:
            return []
            
        features_list = []
        if is_clean:
            # Extract up to 4 random patches
            num_patches = min(4, max(1, (H // 128) * (W // 128)))
            for _ in range(num_patches):
                y = random.randint(0, H - 128)
                x = random.randint(0, W - 128)
                patch = img[y:y+128, x:x+128]
                f = extract_patch_features(patch)
                features_list.append(f)
        else:
            # PVD always embeds in top-left
            patch = img[0:128, 0:128]
            f = extract_patch_features(patch)
            features_list.append(f)
            
        return features_list
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return []

def load_data():
    X = []
    y = []
    
    print("Loading CLEAN patches...")
    for clean_dir in CLEAN_TRAIN_DIRS:
        files = list_image_files(clean_dir)
        for f in files:
            feats_list = extract_training_features(f, is_clean=True)
            for feat in feats_list:
                X.append(feat)
                y.append(0) # 0 for CLEAN

    print("Loading STEGO patches...")
    for stego_dir in STEGO_TRAIN_DIRS:
        files = list_image_files(stego_dir)
        for f in files:
            feats_list = extract_training_features(f, is_clean=False)
            for feat in feats_list:
                X.append(feat)
                y.append(1) # 1 for STEGO
                
    return np.array(X), np.array(y)

def main():
    print("--- Starting Training Pipeline ---")
    
    # 1. Load Data
    X, y = load_data()
    print(f"\nLoaded {len(X)} total samples.")
    print(f"  Clean instances: {np.sum(y == 0)}")
    print(f"  Stego instances: {np.sum(y == 1)}")
    
    if len(X) == 0:
        print("ERROR: No data loaded. Check the dataset paths.")
        return
        
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Feature Scaling (Crucial for generalization)
    print("\nFitting StandardScaler to training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Model
    print("Training RandomForest Classifier...")
    # Using specific params to prevent overfitting while capturing the data nuances
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # 5. Evaluate Model
    print("\n--- Evaluation on Held-Out Test Set ---")
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['CLEAN', 'STEGO']))
    
    # 6. Save Model and Scaler
    print(f"\nSaving model to: {MODEL_PATH}")
    joblib.dump(clf, MODEL_PATH)
    
    print(f"Saving scaler to: {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    
    print("\n✅ Training Complete. Model and Scaler are ready for use.")

if __name__ == '__main__':
    main()
