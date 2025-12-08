"""
Script training mô hình
Hỗ trợ nhiều loại mô hình khác nhau
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score)
import joblib
import json
from pathlib import Path
from datetime import datetime


class ModelTrainer:
    """Class training và tuning mô hình"""
    
    def __init__(self, model_type='classification', model_name='random_forest'):
        """
        Parameters:
        -----------
        model_type : str
            'classification' hoặc 'regression'
        model_name : str
            'random_forest', 'svm', 'logistic', 'linear'
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.best_params = None
        self.training_history = {}
    
    def _get_base_model(self):
        """Lấy base model chưa tune"""
        if self.model_type == 'classification':
            if self.model_name == 'random_forest':
                return RandomForestClassifier(random_state=42)
            elif self.model_name == 'svm':
                return SVC(random_state=42)
            elif self.model_name == 'logistic':
                return LogisticRegression(random_state=42, max_iter=1000)
        else:  # regression
            if self.model_name == 'random_forest':
                return RandomForestRegressor(random_state=42)
            elif self.model_name == 'svm':
                return SVR()
            elif self.model_name == 'linear':
                return LinearRegression()
        return RandomForestClassifier(random_state=42) if self.model_type == 'classification' else RandomForestRegressor(random_state=42)
    
    def _get_param_grid(self):
        """Lấy parameter grid cho tuning"""
        if self.model_name == 'random_forest':
            if self.model_type == 'classification':
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
        elif self.model_name == 'svm':
            return {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
        elif self.model_name == 'logistic':
            return {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif self.model_name == 'linear':
            return {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        return {}
    
    def train(self, X_train, y_train, tuning_method='grid', cv=5, n_iter=50):
        """
        Training mô hình với hyperparameter tuning
        
        Parameters:
        -----------
        X_train : pd.DataFrame hoặc np.array
        y_train : pd.Series hoặc np.array
        tuning_method : str
            'grid' hoặc 'random'
        cv : int
            Số fold cho cross-validation
        n_iter : int
            Số iteration cho RandomizedSearchCV
        """
        base_model = self._get_base_model()
        param_grid = self._get_param_grid()
        
        if tuning_method == 'grid' and len(param_grid) > 0:
            search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy' if self.model_type == 'classification' else 'r2', n_jobs=-1)
        elif tuning_method == 'random' and len(param_grid) > 0:
            search = RandomizedSearchCV(base_model, param_grid, cv=cv, n_iter=n_iter, 
                                       scoring='accuracy' if self.model_type == 'classification' else 'r2', n_jobs=-1, random_state=42)
        else:
            # Không tuning, train trực tiếp
            self.model = base_model
            self.model.fit(X_train, y_train)
            return self.model
        
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.training_history = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_scores': search.cv_results_
        }
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình"""
        y_pred = self.model.predict(X_test)
        
        if self.model_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            metrics = {
                'rmse': np.sqrt(mse),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100,
                'r2': r2_score(y_test, y_pred)
            }
        
        return metrics, y_pred
    
    def save_model(self, filepath: str = None, metadata: dict = None):
        """Lưu mô hình và metadata"""
        if filepath is None:
            filepath = f"src/models/{self.model_name}_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu model
        joblib.dump(self.model, filepath)
        
        # Lưu metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_type': self.model_type,
            'model_name': self.model_name,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat()
        })
        
        metadata_path = filepath.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return filepath, metadata_path
    
    @staticmethod
    def load_model(filepath: str):
        """Load mô hình đã lưu"""
        return joblib.load(filepath)

