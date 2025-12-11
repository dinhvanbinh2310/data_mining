# Đồ án Khai thác Dữ liệu

## Mô tả
Project khai thác dữ liệu với quy trình đầy đủ từ tiền xử lý đến đánh giá mô hình.

## Cấu trúc Project

```
project/
├── data/
│   ├── raw/              # Dữ liệu thô
│   └── processed/        # Dữ liệu đã xử lý
├── src/
│   ├── preprocess/       # Code tiền xử lý
│   ├── models/           # Mô hình ML
│   ├── evaluation/       # Đánh giá mô hình
│   ├── utils/            # Utility functions
│   ├── app/              # Demo app (optional)
│   └── main.ipynb        # Notebook chính
├── report/
│   ├── draft/            # Báo cáo nháp
│   └── final/            # Báo cáo cuối
├── slides/               # Slides thuyết trình
└── environment/          # Cài đặt môi trường
```

## Dataset

**Dataset hiện tại**: Student Performance (UCI ML Repository)
- **Nguồn**: https://archive.ics.uci.edu/dataset/320/student+performance
- **Số dòng**: 649
- **Số cột**: 33
- **Loại bài toán**: Regression (dự đoán điểm cuối kỳ G3)
- **File**: `data/raw/student.csv`

### Tải Dataset

Dataset đã được tải sẵn trong `data/raw/student.csv`. Nếu cần tải lại:

```bash
python src/utils/download_dataset.py
```

Hoặc sử dụng ucimlrepo trong Python:
```python
from ucimlrepo import fetch_ucirepo
student_performance = fetch_ucirepo(id=320)
```

## Yêu cầu Dataset

- Tối thiểu 5 cột và 500 dòng
- Có thông tin nguồn dữ liệu
- Đặt file dataset vào `data/raw/`

## Quy trình

1. **Kiểm tra Dataset**: Thống kê mô tả, phân tích dữ liệu
2. **Tiền xử lý**: Xử lý missing, outlier, encoding, scaling
3. **Xây dựng Mô hình**: 2 mô hình khác loại với hyperparameter tuning
4. **Đánh giá**: Metrics phù hợp với loại bài toán
5. **Báo cáo**: Tạo PDF báo cáo đầy đủ

## Cài đặt

Xem chi tiết tại [environment/install.md](environment/install.md)

```bash
pip install -r environment/requirements.txt
```

## Sử dụng

### Pipeline chính

1. Đặt dataset vào `data/raw/`
2. Chạy `src/main.ipynb` để thực hiện toàn bộ pipeline
3. Xem kết quả tại `src/evaluation/` và `report/final/`

### Demo App

Chạy Streamlit app để dự đoán điểm học tập:

**Cách 1: Dùng script helper (khuyến nghị)**
```bash
# Trên Windows (Git Bash hoặc CMD)
./run_app.sh
# hoặc
run_app.bat

# Trên Linux/Mac
./run_app.sh
```

**Cách 2: Chạy thủ công**
```bash
# Kích hoạt virtual environment
source .venv/Scripts/activate  # Git Bash
# hoặc
.venv\Scripts\activate.bat     # CMD/PowerShell

# Chạy app
streamlit run src/app/app.py
```

App sẽ mở tại `http://localhost:8501`

**Tính năng:**
- Chọn model (Random Forest hoặc SVM)
- Nhập thông tin học sinh qua form
- Dự đoán điểm cuối kỳ (G3)
- Xem đánh giá và kết quả

Xem chi tiết tại [src/app/README.md](src/app/README.md)

## Tác giả
[Điền thông tin tác giả]

# data_mining
