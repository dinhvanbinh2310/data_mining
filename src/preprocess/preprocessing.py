"""
Module tiền xử lý dữ liệu
Xử lý missing values, outliers, encoding, scaling
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib


class DataPreprocessor:
    """Class xử lý tiền xử lý dữ liệu"""
    
    def __init__(self, 
                 missing_strategy='mean',  # mean, median, most_frequent, interpolation
                 outlier_method='IQR',      # IQR, Z-score
                 encoding_method='onehot',  # onehot, label, ordinal
                 scaling_method='standard'  # standard, minmax
                 ):
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.encoding_method = encoding_method
        self.scaling_method = scaling_method
        
        self.scalers = {}
        self.encoders = {}
        self.preprocessing_info = {}
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý giá trị thiếu"""
        df_processed = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if self.missing_strategy == 'mean':
            for col in numeric_cols:
                df_processed[col].fillna(df[col].mean(), inplace=True)
        elif self.missing_strategy == 'median':
            for col in numeric_cols:
                df_processed[col].fillna(df[col].median(), inplace=True)
        elif self.missing_strategy == 'most_frequent':
            for col in numeric_cols:
                df_processed[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0, inplace=True)
            for col in categorical_cols:
                df_processed[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown', inplace=True)
        elif self.missing_strategy == 'interpolation':
            for col in numeric_cols:
                df_processed[col].interpolate(method='linear', inplace=True)
        
        self.preprocessing_info['missing_handled'] = True
        return df_processed
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý outliers"""
        df_processed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if self.outlier_method == 'IQR':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif self.outlier_method == 'Z-score':
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df_processed = df_processed[z_scores < 3]
        
        self.preprocessing_info['outliers_handled'] = True
        return df_processed
    
    def encode_features(self, df: pd.DataFrame, fit=True) -> pd.DataFrame:
        """Encoding các features phân loại"""
        df_processed = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return df_processed
        
        if self.encoding_method == 'onehot':
            if fit:
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[categorical_cols])
                self.encoders['onehot'] = encoder
            else:
                encoded = self.encoders['onehot'].transform(df[categorical_cols])
            
            # Tạo tên cột
            feature_names = self.encoders['onehot'].get_feature_names_out(categorical_cols)
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            df_processed = pd.concat([df_processed.drop(categorical_cols, axis=1), encoded_df], axis=1)
        
        elif self.encoding_method == 'label':
            for col in categorical_cols:
                if fit:
                    encoder = LabelEncoder()
                    df_processed[col] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[f'label_{col}'] = encoder
                else:
                    df_processed[col] = self.encoders[f'label_{col}'].transform(df[col].astype(str))
        
        elif self.encoding_method == 'ordinal':
            if fit:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                df_processed[categorical_cols] = encoder.fit_transform(df[categorical_cols])
                self.encoders['ordinal'] = encoder
            else:
                df_processed[categorical_cols] = self.encoders['ordinal'].transform(df[categorical_cols])
        
        self.preprocessing_info['encoding_done'] = True
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, fit=True) -> pd.DataFrame:
        """Scaling features"""
        df_processed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df_processed
        
        if self.scaling_method == 'standard':
            if fit:
                scaler = StandardScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                self.scalers['standard'] = scaler
            else:
                df_processed[numeric_cols] = self.scalers['standard'].transform(df[numeric_cols])
        
        elif self.scaling_method == 'minmax':
            if fit:
                scaler = MinMaxScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                self.scalers['minmax'] = scaler
            else:
                df_processed[numeric_cols] = self.scalers['minmax'].transform(df[numeric_cols])
        
        self.preprocessing_info['scaling_done'] = True
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thực hiện toàn bộ pipeline tiền xử lý (fit + transform)"""
        df_processed = df.copy()
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.handle_outliers(df_processed)
        df_processed = self.encode_features(df_processed, fit=True)
        df_processed = self.scale_features(df_processed, fit=True)
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dữ liệu mới (không fit)"""
        df_processed = df.copy()
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.handle_outliers(df_processed)
        df_processed = self.encode_features(df_processed, fit=False)
        df_processed = self.scale_features(df_processed, fit=False)
        return df_processed
    
    def save_preprocessor(self, filepath: str = "src/models/preprocessor.joblib"):
        """Lưu preprocessor"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
    
    @staticmethod
    def load_preprocessor(filepath: str = "src/models/preprocessor.joblib"):
        """Load preprocessor"""
        return joblib.load(filepath)


def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, stratify=None, random_state=42):
    """Chia train/test set"""
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)

