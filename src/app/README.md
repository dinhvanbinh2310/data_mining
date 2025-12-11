# Demo App - Dự đoán Điểm Học tập

Streamlit app để dự đoán điểm cuối kỳ (G3) của học sinh dựa trên thông tin cá nhân, gia đình, học tập và xã hội.

## Cài đặt

```bash
pip install streamlit
```

Hoặc cài tất cả dependencies:

```bash
pip install -r environment/requirements.txt
```

## Chạy App

```bash
streamlit run src/app/app.py
```

App sẽ mở tự động trong trình duyệt tại `http://localhost:8501`

## Tính năng

- ✅ Chọn model (Model A: Random Forest hoặc Model B: SVM)
- ✅ Form nhập liệu đầy đủ với 30 features
- ✅ Dự đoán điểm G3 (0-20)
- ✅ Hiển thị đánh giá kết quả
- ✅ Xem lại thông tin đã nhập

## Yêu cầu

- Models đã được train và lưu tại `src/models/`
- Preprocessor đã được lưu tại `src/models/preprocessor.joblib`

