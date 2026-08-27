"""
Evaluation metrics module for ASVspoof Audio Deepfake Detection.
Calculates Equal Error Rate (EER) and optimal threshold.
"""

import numpy as np

def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray):
    """
    Computes Equal Error Rate (EER) and threshold from bonafide (real) and spoof (fake) scores.
    Score convention: Higher score = more likely bonafide (real).
    
    Returns:
        eer (float): Equal Error Rate as a percentage (e.g. 0.015 for 1.5%).
        threshold (float): Decision threshold at EER.
    """
    bonafide_scores = np.asarray(bonafide_scores)
    spoof_scores = np.asarray(spoof_scores)
    
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 0.0
        
    # All unique thresholds across both score distributions
    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    thresholds = np.sort(np.unique(all_scores))
    
    # Calculate False Rejection Rate (FRR): bonafide rejected (< threshold)
    # Calculate False Acceptance Rate (FAR): spoof accepted (>= threshold)
    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)
    
    frr = np.searchsorted(np.sort(bonafide_scores), thresholds, side='left') / float(n_bonafide)
    far = (n_spoof - np.searchsorted(np.sort(spoof_scores), thresholds, side='left')) / float(n_spoof)
    
    # Find index where FAR and FRR cross
    diff = far - frr
    min_idx = np.argmin(np.abs(diff))
    
    if min_idx < len(thresholds) - 1 and diff[min_idx] * diff[min_idx + 1] < 0:
        # Linear interpolation between adjacent points
        eer = (frr[min_idx] + frr[min_idx + 1] + far[min_idx] + far[min_idx + 1]) / 4.0
        threshold = (thresholds[min_idx] + thresholds[min_idx + 1]) / 2.0
    else:
        eer = (frr[min_idx] + far[min_idx]) / 2.0
        threshold = thresholds[min_idx]
        
    return float(eer), float(threshold)

def compute_eer_from_probabilities(y_true: np.ndarray, p_fake: np.ndarray):
    """
    Computes EER given ground truth labels y_true (0=real, 1=fake) and P(fake) predictions.
    Bonafide score = 1.0 - P(fake) = P(real).
    """
    y_true = np.asarray(y_true)
    p_fake = np.asarray(p_fake)
    
    bonafide_scores = 1.0 - p_fake[y_true == 0]
    spoof_scores = 1.0 - p_fake[y_true == 1]
    
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    # compute_eer returns the threshold for the bonafide_score (1.0 - p_fake)
    # We want to return the threshold for p_fake
    return eer, 1.0 - threshold
