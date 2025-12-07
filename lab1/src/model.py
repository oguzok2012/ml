import numpy as np
import warnings
from src.metrics import mse

class MyLinearRegression:
    def __init__(
        self,
        method="analytic",
        lr=0.01,           # Learning Rate
        max_iter=1000,
        tol=1e-5,
        random_state=None,
        reg_type=None,     # 'l1', 'l2', 'elasticnet' (только для gd/sgd) или None
        reg_alpha=0.0,     # Коэффициент регуляризации
        l1_ratio=0.5       # Для ElasticNet
    ):
        self.method = method
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.reg_type = reg_type
        self.reg_alpha = reg_alpha
        self.l1_ratio = l1_ratio
        self.weights = None
        self.mse_history = []

    def _add_bias(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X_b = self._add_bias(X)
        n_features = X_b.shape[1]

        if self.method == "analytic":
            self._fit_analytic(X_b, y, n_features)
        elif self.method == "gd":
            self._fit_gd(X_b, y, n_features)
        elif self.method == "sgd":
            self._fit_sgd(X_b, y, n_features)
        
        return self

    def _fit_analytic(self, X, y, n_features):
        # Аналитическое решение возможно ТОЛЬКО для OLS и Ridge (L2)
        # Для Lasso (L1) аналитической формулы нет (нужны итеративные методы)
        
        I = np.eye(n_features)
        I[0, 0] = 0  # Bias не штрафуем
        
        if self.reg_type == 'l1':
            warnings.warn("Analytic method does not support L1 (Lasso). Switching to OLS (alpha=0).")
            alpha = 0
        elif self.reg_type == 'l2':
            alpha = self.reg_alpha
        else:
            alpha = 0
        
        # w = (X^T X + alpha*I)^-1 X^T y
        matrix = X.T @ X + alpha * I
        try:
            self.weights = np.linalg.inv(matrix) @ X.T @ y
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(matrix) @ X.T @ y
            
        self.mse_history = [mse(y, X @ self.weights)]

    def _fit_gd(self, X, y, n_features):
        n_samples = X.shape[0]
        self.weights = np.zeros(n_features)
        prev_error = float('inf')

        for i in range(self.max_iter):
            y_pred = X @ self.weights
            
            # Основной градиент MSE
            grad = (2.0 / n_samples) * X.T @ (y_pred - y)
            
            # Добавка регуляризации
            if self.reg_type == 'l2':
                reg_grad = 2 * self.reg_alpha * self.weights
                reg_grad[0] = 0
                grad += reg_grad
            elif self.reg_type == 'l1':
                reg_grad = self.reg_alpha * np.sign(self.weights)
                reg_grad[0] = 0
                grad += reg_grad
            elif self.reg_type == 'elasticnet':
                l1 = self.l1_ratio * self.reg_alpha * np.sign(self.weights)
                l2 = (1 - self.l1_ratio) * 2 * self.reg_alpha * self.weights
                reg_grad = l1 + l2
                reg_grad[0] = 0
                grad += reg_grad

            # Шаг спуска
            self.weights -= self.lr * grad
            
            # Проверка на взрыв
            if not np.all(np.isfinite(self.weights)):
                print(f"Warning: GD diverged at iteration {i}. Try reducing Learning Rate.")
                break

            curr_error = np.mean((y - X @ self.weights)**2)
            self.mse_history.append(curr_error)
            
            if abs(prev_error - curr_error) < self.tol:
                break
            prev_error = curr_error

    def _fit_sgd(self, X, y, n_features):
        n_samples = X.shape[0]
        self.weights = np.zeros(n_features)
        
        if self.random_state:
            np.random.seed(self.random_state)
            
        lr = self.lr

        for epoch in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            epoch_error = 0.0
            
            for idx in indices:
                xi = X[idx] 
                yi = y[idx] 
                
                y_pred = np.dot(xi, self.weights)
                error = y_pred - yi
                
                # Градиент для одного примера
                grad = 2.0 * xi * error
                
                # Регуляризация
                if self.reg_type == 'l2':
                    reg_grad = 2 * self.reg_alpha * self.weights / n_samples # Нормируем на N для SGD
                    reg_grad[0] = 0
                    grad += reg_grad
                elif self.reg_type == 'l1':
                    reg_grad = self.reg_alpha * np.sign(self.weights) / n_samples
                    reg_grad[0] = 0
                    grad += reg_grad
                
                # Обновление
                self.weights -= lr * grad
                epoch_error += error ** 2
            
            # Проверка на взрыв
            if not np.all(np.isfinite(self.weights)):
                print(f"Warning: SGD diverged at epoch {epoch}. Try reducing Learning Rate.")
                break
            
            self.mse_history.append(epoch_error / n_samples)
            
            # Затухание LR
            if epoch > 0 and epoch % 100 == 0:
                lr *= 0.9

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_b = self._add_bias(X)
        preds = X_b @ self.weights
        # Защита от NaN в выводе (заменяем на 0 или среднее, чтобы sklearn не падал)
        if not np.all(np.isfinite(preds)):
            return np.zeros_like(preds)
        return preds