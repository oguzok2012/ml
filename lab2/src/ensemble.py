import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class MyBaggingClassifier:
    def __init__(self, n_estimators=10, max_samples=1.0, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        
    def fit(self, X, y):
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(y, "values") else y
        
        np.random.seed(self.random_state)
        n_samples = int(len(X_arr) * self.max_samples)
        
        self.estimators_ = []
        for i in range(self.n_estimators):
            indices = np.random.choice(len(X_arr), n_samples, replace=True)
            X_sample = X_arr[indices]
            y_sample = y_arr[indices]
            
            tree_seed = (self.random_state + i) if self.random_state is not None else None
            
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=tree_seed)
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)
        return self
    
    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, 'values') else X
        probas = np.array([tree.predict_proba(X_arr) for tree in self.estimators_])
        return np.mean(probas, axis=0)[:, 1]
    
    def predict(self, X):
        X_arr = X.values if hasattr(X, 'values') else X
        return (self.predict_proba(X_arr) >= 0.5).astype(int)


class MyGradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 subsample=1.0, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.estimators_ = []
        self.init_pred = None
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X, y):
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(y, "values") else y
        
        np.random.seed(self.random_state)
        
        pos_rate = np.mean(y_arr)
        self.init_pred = np.log(pos_rate / (1 - pos_rate + 1e-15))
        
        F = np.full(len(y_arr), self.init_pred, dtype=np.float64)
        
        for i in range(self.n_estimators):
            p = self._sigmoid(F)
            
            residuals = y_arr - p
            
            if self.subsample < 1.0:
                n_samples = int(len(X_arr) * self.subsample)
                indices = np.random.choice(len(X_arr), n_samples, replace=False)
                X_subset, r_subset = X_arr[indices], residuals[indices]
                p_subset = p[indices] 
            else:
                X_subset, r_subset = X_arr, residuals
                p_subset = p
            
            tree_seed = (self.random_state + i) if self.random_state is not None else None
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed
            )
            tree.fit(X_subset, r_subset)
            
            leaf_indices = tree.apply(X_subset)
            unique_leaves = np.unique(leaf_indices)
            
            hessian = p_subset * (1 - p_subset)
            
            for leaf in unique_leaves:
                mask = leaf_indices == leaf
                
                sum_residuals = np.sum(r_subset[mask])
                sum_hessian = np.sum(hessian[mask])
                
                gamma = sum_residuals / (sum_hessian + 1e-15)
                
                tree.tree_.value[leaf, 0, 0] = gamma
            
            update = tree.predict(X_arr)
            F += self.learning_rate * update
            
            self.estimators_.append(tree)
        
        return self
    
    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, 'values') else X
        F = np.full(len(X_arr), self.init_pred, dtype=np.float64)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X_arr)
        return self._sigmoid(F)
    
    def predict(self, X):
        X_arr = X.values if hasattr(X, 'values') else X
        return (self.predict_proba(X_arr) >= 0.5).astype(int)
