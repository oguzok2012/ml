import numpy as np
from metrics import mse

def my_cross_val_score(model_cls, X, y, cv=5, method='analytic', **kwargs):
    """
    K-Fold Cross Validation.
    model_cls: класс модели (не инстанс, чтобы сбрасывать веса)
    """
    X = np.array(X)
    y = np.array(y)
    n_samples = len(y)
    fold_size = n_samples // cv
    indices = np.arange(n_samples)
    scores = []
    
    for i in range(cv):
        # Индексы для валидации и обучения
        val_idx = indices[i*fold_size : (i+1)*fold_size]
        train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Обучаем новую модель
        model = model_cls(method=method, **kwargs)
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        
        scores.append(mse(y_fold_val, y_pred))
        
    return np.mean(scores), scores

def leave_one_out_cv(model_cls, X, y, method='analytic', max_samples=None, **kwargs):
    """
    Leave-One-Out CV. 
    max_samples: ограничение для демонстрации (иначе очень долго для 10k строк)
    """
    X = np.array(X)
    y = np.array(y)
    n_samples = len(y)
    scores = []
    
    limit = n_samples if max_samples is None else max_samples
    
    print(f"Запуск LOO на {limit} примерах...")
    
    for i in range(limit):
        val_idx = [i]
        train_idx = np.delete(np.arange(n_samples), i)
        
        X_loo_train, X_loo_val = X[train_idx], X[val_idx]
        y_loo_train, y_loo_val = y[train_idx], y[val_idx]
        
        model = model_cls(method=method, **kwargs)
        model.fit(X_loo_train, y_loo_train)
        y_pred = model.predict(X_loo_val)
        
        scores.append((y_loo_val[0] - y_pred[0])**2)
        
    return np.mean(scores)