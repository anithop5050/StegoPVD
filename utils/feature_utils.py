"""
Feature extraction utilities for steganography detection.
Provides patch extraction and probability calculation functions.
"""

import cv2
import numpy as np
from typing import List, Tuple, Iterator
from . import config


def extract_patches(img: np.ndarray, patch_size: int = None) -> List[np.ndarray]:
    """
    Extract non-overlapping patches from an image.
    
    Args:
        img: Input image (H, W, C)
        patch_size: Size of patches (default from config.PATCH_SIZE)
        
    Returns:
        List of patch arrays
    """
    if patch_size is None:
        patch_size = config.PATCH_SIZE
    
    patches = []
    H, W = img.shape[:2]
    
    # Handle small images
    if H < patch_size or W < patch_size:
        # Pad or return single patch
        patches.append(img)
        return patches
    
    # Extract non-overlapping patches
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]
            if patch.shape == (patch_size, patch_size, 3):
                patches.append(patch)
    
    return patches


def iter_patches_with_coords(img: np.ndarray, patch_size: int = None) -> Iterator[Tuple[np.ndarray, int, int]]:
    """
    Iterate over image patches with their coordinates.
    
    Args:
        img: Input image (H, W, C)
        patch_size: Size of patches (default from config.PATCH_SIZE)
        
    Yields:
        Tuple of (patch, y_coord, x_coord)
    """
    if patch_size is None:
        patch_size = config.PATCH_SIZE
    
    H, W = img.shape[:2]
    
    # Handle small images
    if H < patch_size or W < patch_size:
        yield (img, 0, 0)
        return
    
    # Extract non-overlapping patches with coordinates
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]
            if patch.shape == (patch_size, patch_size, 3):
                yield (patch, y, x)


def extract_histogram_features(patch: np.ndarray) -> np.ndarray:
    """
    Extract histogram-based features from a patch.
    Used by classical RandomForest model.
    
    Args:
        patch: Image patch (H, W, C)
        
    Returns:
        Feature vector (1280-D for classical model)
    """
    features = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    # Histogram features from different channels
    for channel in [0, 1, 2]:  # B, G, R
        hist = cv2.calcHist([patch], [channel], None, [256], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)  # Normalize
        features.extend(hist)
    
    # Grayscale histogram
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_gray = hist_gray.flatten() / (hist_gray.sum() + 1e-7)
    features.extend(hist_gray)
    
    # Edge histogram (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    hist_edges = cv2.calcHist([edges], [0], None, [256], [0, 256])
    hist_edges = hist_edges.flatten() / (hist_edges.sum() + 1e-7)
    features.extend(hist_edges)
    
    return np.array(features, dtype=np.float32)


def extract_patch_probabilities(img: np.ndarray, model, scaler) -> List[float]:
    """
    Extract stego probabilities for all patches using classical model.
    
    Args:
        img: Input image (H, W, C)
        model: Trained classifier (e.g., RandomForest)
        scaler: Feature scaler (e.g., StandardScaler)
        
    Returns:
        List of stego probabilities for each patch
    """
    patch_probs = []
    
    for patch, y, x in iter_patches_with_coords(img):
        # Extract features
        features = extract_histogram_features(patch)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict probability
        probs = model.predict_proba(features_scaled)
        stego_prob = probs[0, 1]  # Probability of stego class
        
        patch_probs.append(float(stego_prob))
    
    return patch_probs


def preprocess_image(img: np.ndarray, max_dimension: int = 2048) -> Tuple[np.ndarray, bool]:
    """
    Preprocess image for inference: resize if too large, validate format.
    
    Args:
        img: Input image
        max_dimension: Maximum allowed dimension
        
    Returns:
        Tuple of (preprocessed_image, was_resized)
    """
    if img is None or img.size == 0:
        raise ValueError("Image is None or empty")
    
    H, W = img.shape[:2]
    was_resized = False
    
    # Resize if too large
    if max(H, W) > max_dimension:
        scale = max_dimension / max(H, W)
        new_W = int(W * scale)
        new_H = int(H * scale)
        img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
        was_resized = True
    
    # Ensure 3 channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) != 3 or img.shape[2] != 3:
        # Handle unexpected image formats
        if len(img.shape) == 3:
            # Take first 3 channels if somehow has more
            img = img[:, :, :3]
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
    
    # Final validation
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Final image validation failed: {img.shape}")
    
    return img, was_resized
