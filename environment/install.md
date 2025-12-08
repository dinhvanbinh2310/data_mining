# Hướng dẫn Cài đặt Môi trường

## Yêu cầu hệ thống
- Python >= 3.9
- pip hoặc conda

## Cài đặt với pip

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r environment/requirements.txt
```

## Cài đặt với conda

```bash
# Tạo môi trường conda
conda create -n data_mining python=3.9
conda activate data_mining

# Cài đặt dependencies
pip install -r environment/requirements.txt
```

## Kiểm tra cài đặt

```bash
python -c "import pandas, numpy, sklearn; print('OK')"
```

## Cấu trúc thư mục

Sau khi cài đặt, đảm bảo có đủ các thư mục:
- `data/raw/` - Dữ liệu thô
- `data/processed/` - Dữ liệu đã xử lý
- `src/` - Source code
- `report/` - Báo cáo
- `slides/` - Slides thuyết trình

