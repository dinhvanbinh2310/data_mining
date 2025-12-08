"""
Utility functions cho thống kê mô tả dữ liệu
"""
import pandas as pd
import numpy as np
from pathlib import Path


def generate_descriptive_stats(df: pd.DataFrame, output_dir: str = "report") -> dict:
    """
    Tạo thống kê mô tả tự động cho dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset cần phân tích
    output_dir : str
        Thư mục lưu kết quả
        
    Returns:
    --------
    dict : Dictionary chứa các thống kê
    """
    stats = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_count": df.isnull().sum().to_dict(),
        "missing_rate": (df.isnull().sum() / len(df) * 100).to_dict(),
        "numeric_stats": {},
        "categorical_stats": {}
    }
    
    # Thống kê cho cột số
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats["numeric_stats"] = df[numeric_cols].describe().to_dict()
        stats["numeric_stats"]["median"] = df[numeric_cols].median().to_dict()
    
    # Thống kê cho cột phân loại
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        stats["categorical_stats"][col] = {
            "unique_count": df[col].nunique(),
            "top_values": df[col].value_counts().head(10).to_dict(),
            "frequency": df[col].value_counts(normalize=True).head(10).to_dict()
        }
    
    return stats


def save_stats_to_markdown(stats: dict, output_path: str = "report/descriptive_stats.md"):
    """
    Lưu thống kê ra file markdown
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Thống kê Mô tả Dataset\n\n")
        f.write(f"## Tổng quan\n")
        f.write(f"- Số dòng: {stats['shape'][0]}\n")
        f.write(f"- Số cột: {stats['shape'][1]}\n\n")
        
        f.write(f"## Thông tin cột\n")
        for col, dtype in stats['dtypes'].items():
            missing = stats['missing_count'][col]
            missing_pct = stats['missing_rate'][col]
            f.write(f"- **{col}** ({dtype}): {missing} missing ({missing_pct:.2f}%)\n")
        f.write("\n")
        
        if stats['numeric_stats']:
            f.write("## Thống kê số\n")
            for col in stats['numeric_stats'].get('mean', {}).keys():
                f.write(f"### {col}\n")
                f.write(f"- Mean: {stats['numeric_stats']['mean'][col]:.2f}\n")
                f.write(f"- Median: {stats['numeric_stats']['median'][col]:.2f}\n")
                f.write(f"- Min: {stats['numeric_stats']['min'][col]:.2f}\n")
                f.write(f"- Max: {stats['numeric_stats']['max'][col]:.2f}\n")
                f.write(f"- Std: {stats['numeric_stats']['std'][col]:.2f}\n\n")
        
        if stats['categorical_stats']:
            f.write("## Thống kê phân loại\n")
            for col, cat_stats in stats['categorical_stats'].items():
                f.write(f"### {col}\n")
                f.write(f"- Số giá trị unique: {cat_stats['unique_count']}\n")
                f.write(f"- Top values:\n")
                for val, count in list(cat_stats['top_values'].items())[:5]:
                    f.write(f"  - {val}: {count}\n")
                f.write("\n")


def save_stats_to_csv(df: pd.DataFrame, output_path: str = "src/utils/descriptive_stats.csv"):
    """
    Lưu thống kê cơ bản ra CSV
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    summary = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.values,
        'missing_count': df.isnull().sum().values,
        'missing_rate': (df.isnull().sum() / len(df) * 100).values,
        'unique_count': [df[col].nunique() for col in df.columns]
    })
    
    summary.to_csv(output_path, index=False, encoding='utf-8')
    return summary

