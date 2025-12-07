
import numpy as np

def z_score_normalize(X):
    X = np.asarray(X, dtype=np.float64)
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std

def min_max_normalize(X):
    X = np.asarray(X, dtype=np.float64)
    X_min = np.min(X, axis=0, keepdims=True)
    X_max = np.max(X, axis=0, keepdims=True)
    denom = X_max - X_min
    denom = np.where(denom == 0, 1.0, denom)
    return (X - X_min) / denom