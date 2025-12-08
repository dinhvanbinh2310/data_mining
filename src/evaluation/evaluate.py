"""
Module đánh giá mô hình
Tính các metrics và tạo visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report,
                           mean_squared_error, mean_absolute_error, r2_score,
                           silhouette_score, davies_bouldin_score)
from pathlib import Path
import json
from datetime import datetime


class ModelEvaluator:
    """Class đánh giá mô hình"""
    
    def __init__(self, model_type='classification'):
        """
        Parameters:
        -----------
        model_type : str
            'classification', 'regression', hoặc 'clustering'
        """
        self.model_type = model_type
        self.metrics = {}
        self.figures = []
    
    def evaluate_classification(self, y_true, y_pred, save_path: str = None):
        """Đánh giá mô hình classification"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Confusion Matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Metrics bar chart
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        sns.barplot(data=metrics_df, x='Metric', y='Value', ax=axes[1])
        axes[1].set_title('Classification Metrics')
        axes[1].set_ylim(0, 1)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu hình vào: {save_path}")
        
        self.metrics = metrics
        self.metrics['confusion_matrix'] = cm.tolist()
        
        return metrics, fig
    
    def evaluate_regression(self, y_true, y_pred, save_path: str = None):
        """Đánh giá mô hình regression"""
        mse = mean_squared_error(y_true, y_pred)
        metrics = {
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
            'r2': r2_score(y_true, y_pred)
        }
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot: predicted vs actual
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Predicted vs Actual')
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu hình vào: {save_path}")
        
        self.metrics = metrics
        return metrics, fig
    
    def evaluate_clustering(self, X, labels, save_path: str = None):
        """Đánh giá mô hình clustering"""
        metrics = {
            'silhouette_score': silhouette_score(X, labels),
            'davies_bouldin_score': davies_bouldin_score(X, labels)
        }
        
        # Visualization (nếu 2D hoặc có thể giảm chiều)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        if X.shape[1] == 2:
            # 2D scatter
            scatter = axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
            axes[0].set_title('Cluster Visualization')
            plt.colorbar(scatter, ax=axes[0])
        else:
            # PCA để visualize
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
            axes[0].set_title('Cluster Visualization (PCA)')
            plt.colorbar(scatter, ax=axes[0])
        
        # Metrics bar chart
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        sns.barplot(data=metrics_df, x='Metric', y='Value', ax=axes[1])
        axes[1].set_title('Clustering Metrics')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu hình vào: {save_path}")
        
        self.metrics = metrics
        return metrics, fig
    
    def compare_models(self, results: dict, save_path: str = None):
        """
        So sánh nhiều mô hình
        
        Parameters:
        -----------
        results : dict
            {'model_name': metrics_dict}
        """
        # Convert to DataFrame
        comparison_df = pd.DataFrame(results).T
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.plot(kind='bar', ax=ax)
        ax.set_title('Model Comparison')
        ax.set_ylabel('Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu hình vào: {save_path}")
        
        return comparison_df, fig
    
    def save_results(self, output_path: str = "src/evaluation/results.json"):
        """Lưu kết quả đánh giá"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'model_type': self.model_type,
            'metrics': self.metrics,
            'evaluated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Đã lưu kết quả vào: {output_path}")

