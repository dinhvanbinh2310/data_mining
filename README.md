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

1. Đặt dataset vào `data/raw/`
2. Chạy `src/main.ipynb` để thực hiện toàn bộ pipeline
3. Xem kết quả tại `src/evaluation/` và `report/final/`

## Tác giả
[Điền thông tin tác giả]

# data_mining
