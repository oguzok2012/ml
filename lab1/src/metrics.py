import numpy as np
from typing import Union, Callable


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    ss_res = np.sum((y_true - y_pred) ** 2)  # residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # total sum of squares
    
    if ss_tot == 0:
        return 0.0  # sklearn behaviour
    return 1.0 - (ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (in %)
    
    Follows sklearn's behaviour (sklearn >= 0.24):
    - Ignores samples where y_true == 0
    - Returns 0.0 if all y_true == 0
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Находим индексы, где y_true != 0
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return 0.0  # как в sklearn
    
    # Берём только ненулевые
    y_true_nz = y_true[non_zero_mask]
    y_pred_nz = y_pred[non_zero_mask]
    
    # Считаем абсолютную относительную ошибку
    ape = np.abs((y_true_nz - y_pred_nz) / y_true_nz)
    
    return 100.0 * np.mean(ape)