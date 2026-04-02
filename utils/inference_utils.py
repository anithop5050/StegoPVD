"""
Inference utilities for aggregating patch predictions.
Implements different aggregation policies for multi-patch predictions.
"""

import numpy as np
from typing import List, Dict
from . import config


def aggregate_patch_decision(patch_probs: List[float]) -> Dict:
    """
    Aggregate patch probabilities into final prediction using configured policy.
    
    Args:
        patch_probs: List of stego probabilities for each patch
        
    Returns:
        Dictionary with prediction result and statistics
    """
    if not patch_probs:
        return {
            'result': 'UNKNOWN',
            'confidence': 0.0,
            'max_prob': 0.0,
            'mean_prob': 0.0,
            'suspicious_ratio': 0.0,
            'suspicious_patches': 0,
            'total_patches': 0
        }
    
    patch_probs = np.array(patch_probs)
    
    # Calculate statistics
    max_prob = float(np.max(patch_probs))
    mean_prob = float(np.mean(patch_probs))
    median_prob = float(np.median(patch_probs))
    
    # Count suspicious patches
    suspicious_mask = patch_probs > config.SUSPICIOUS_PATCH_THRESHOLD
    suspicious_count = int(np.sum(suspicious_mask))
    suspicious_ratio = suspicious_count / len(patch_probs)
    
    # Apply aggregation policy
    policy = config.AGGREGATION_POLICY.lower()
    
    if policy == "max":
        # Legacy: Single suspicious patch flags entire image
        decision_prob = max_prob
        prediction = "STEGO" if decision_prob > config.STEGO_THRESHOLD_MAX else "CLEAN"
        
    elif policy == "mean":
        # Recommended: Average all patches (most stable)
        decision_prob = mean_prob
        prediction = "STEGO" if decision_prob > config.MEAN_PROB_THRESHOLD else "CLEAN"
        
    elif policy == "percentile":
        # Use high percentile (e.g., 90th)
        percentile_value = config.AGGREGATION_PERCENTILE
        decision_prob = float(np.percentile(patch_probs, percentile_value))
        prediction = "STEGO" if decision_prob > config.MEAN_PROB_THRESHOLD else "CLEAN"
        
    elif policy == "voting":
        # Vote based on suspicious patch ratio
        decision_prob = suspicious_ratio
        prediction = "STEGO" if suspicious_ratio > config.STEGO_VOTE_RATIO else "CLEAN"
        
    else:
        # Default to mean
        decision_prob = mean_prob
        prediction = "STEGO" if decision_prob > config.MEAN_PROB_THRESHOLD else "CLEAN"
    
    # Calculate confidence
    if prediction == "STEGO":
        confidence = decision_prob * 100
    else:
        confidence = (1 - decision_prob) * 100
    
    return {
        'result': prediction,
        'confidence': float(confidence),
        'max_prob': float(max_prob),
        'mean_prob': float(mean_prob),
        'median_prob': float(median_prob),
        'suspicious_ratio': float(suspicious_ratio),
        'suspicious_patches': suspicious_count,
        'total_patches': len(patch_probs),
        'aggregation_policy': policy
    }


def aggregate_deep_predictions(patch_probs: List[float], threshold: float = None) -> Dict:
    """
    Aggregate deep model patch predictions using MEAN policy.
    Optimized for deep CNN model output.
    
    Args:
        patch_probs: List of stego probabilities from deep model
        threshold: Decision threshold (default from config)
        
    Returns:
        Dictionary with prediction result and statistics
    """
    if threshold is None:
        threshold = config.MEAN_PROB_THRESHOLD
    
    if not patch_probs:
        return {
            'result': 'UNKNOWN',
            'confidence': 0.0,
            'mean_stego_probability': 0.0,
            'patch_count': 0
        }
    
    patch_probs = np.array(patch_probs)
    
    # Calculate mean probability (as per Experiment 2)
    mean_stego_prob = float(np.mean(patch_probs))
    
    # Apply threshold (as per Experiment 3)
    prediction = "STEGO" if mean_stego_prob > threshold else "CLEAN"
    
    # Confidence is distance from decision boundary
    if prediction == "STEGO":
        confidence = mean_stego_prob * 100
    else:
        confidence = (1 - mean_stego_prob) * 100
    
    return {
        'result': prediction,
        'confidence': float(confidence),
        'mean_stego_probability': float(mean_stego_prob),
        'stego_probability': float(mean_stego_prob),
        'clean_probability': float(1 - mean_stego_prob),
        'patch_count': len(patch_probs),
        'min_prob': float(np.min(patch_probs)),
        'max_prob': float(np.max(patch_probs)),
        'std_prob': float(np.std(patch_probs)),
        'threshold': threshold
    }


def get_confidence_level(confidence: float) -> str:
    """
    Convert numeric confidence to human-readable level.
    
    Args:
        confidence: Confidence percentage (0-100)
        
    Returns:
        Confidence level string
    """
    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Moderate"
    elif confidence >= 50:
        return "Low"
    else:
        return "Very Low"
