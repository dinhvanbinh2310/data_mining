"""
Script predict với mô hình đã train
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path


def load_model_and_preprocessor(model_path: str, preprocessor_path: str = None):
    """
    Load mô hình và preprocessor
    
    Parameters:
    -----------
    model_path : str
        Đường dẫn đến file model (.joblib)
    preprocessor_path : str
        Đường dẫn đến file preprocessor (.joblib)
    """
    model = joblib.load(model_path)
    
    preprocessor = None
    if preprocessor_path and Path(preprocessor_path).exists():
        preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor


def load_metadata(model_path: str):
    """Load metadata của mô hình"""
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    if Path(metadata_path).exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def predict(model, X: pd.DataFrame, preprocessor=None):
    """
    Dự đoán với mô hình
    
    Parameters:
    -----------
    model : trained model
    X : pd.DataFrame
        Dữ liệu đầu vào
    preprocessor : DataPreprocessor
        Preprocessor để transform dữ liệu
        
    Returns:
    --------
    predictions : np.array
    """
    if preprocessor is not None:
        X = preprocessor.transform(X)
    
    predictions = model.predict(X)
    return predictions


def predict_proba(model, X: pd.DataFrame, preprocessor=None):
    """
    Dự đoán xác suất (cho classification)
    """
    if preprocessor is not None:
        X = preprocessor.transform(X)
    
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    return None


def batch_predict(model_path: str, data_path: str, output_path: str = None, 
                  preprocessor_path: str = None):
    """
    Dự đoán hàng loạt từ file CSV
    
    Parameters:
    -----------
    model_path : str
    data_path : str
        Đường dẫn file CSV chứa dữ liệu
    output_path : str
        Đường dẫn lưu kết quả
    preprocessor_path : str
    """
    # Load
    model, preprocessor = load_model_and_preprocessor(model_path, preprocessor_path)
    df = pd.read_csv(data_path)
    
    # Predict
    predictions = predict(model, df, preprocessor)
    
    # Lưu kết quả
    result_df = df.copy()
    result_df['prediction'] = predictions
    
    if output_path is None:
        output_path = data_path.replace('.csv', '_predictions.csv')
    
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Đã lưu kết quả vào: {output_path}")
    
    return result_df


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python predict_model.py <model_path> <data_path> [preprocessor_path]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    preprocessor_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    batch_predict(model_path, data_path, preprocessor_path=preprocessor_path)

