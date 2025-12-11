"""
Script tải dataset Student Performance từ UCI ML Repository
"""
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo

def download_student_performance(output_path="data/raw/student.csv"):
    """Tải dataset Student Performance và lưu vào data/raw/"""
    print("Đang tải dataset Student Performance từ UCI ML Repository...")
    
    try:
        student_performance = fetch_ucirepo(id=320)
        
        X = student_performance.data.features
        y = student_performance.data.targets
        
        df = pd.concat([X, y], axis=1)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        print(f"✅ Đã tải dataset thành công!")
        print(f"   - Số dòng: {len(df)}")
        print(f"   - Số cột: {len(df.columns)}")
        print(f"   - Đã lưu tại: {output_path}")
        
        return df
    
    except Exception as e:
        print(f"❌ Lỗi khi tải dataset: {e}")
        return None

if __name__ == "__main__":
    download_student_performance()

