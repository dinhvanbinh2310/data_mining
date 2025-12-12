# Hướng dẫn Deploy Streamlit App

Có nhiều cách để deploy ứng dụng Streamlit. Dưới đây là các phương án phổ biến:

## 🚀 Phương án 1: Streamlit Cloud (Khuyến nghị - Miễn phí, dễ nhất)

### Bước 1: Đẩy code lên GitHub
```bash
# Khởi tạo git (nếu chưa có)
git init
git add .
git commit -m "Initial commit"

# Tạo repo trên GitHub, sau đó:
git remote add origin https://github.com/username/repo-name.git
git push -u origin main
```

### Bước 2: Deploy trên Streamlit Cloud
1. Truy cập: https://share.streamlit.io/
2. Đăng nhập bằng GitHub
3. Click "New app"
4. Chọn:
   - **Repository**: repo của bạn
   - **Branch**: main (hoặc branch bạn muốn)
   - **Main file path**: `src/app/app.py`
5. Click "Deploy"

### Bước 3: Kiểm tra
- App sẽ tự động deploy và có URL dạng: `https://your-app-name.streamlit.app`
- Mỗi lần push code mới, app sẽ tự động update

### Lưu ý:
- ✅ Miễn phí
- ✅ Tự động deploy khi push code
- ✅ Không cần cấu hình server
- ⚠️ **Models phải có trong repo** (file `.joblib` có thể lớn)
  - Nếu models bị `.gitignore`, cần force add hoặc sửa `.gitignore`
  - Có thể dùng Git LFS cho file lớn: `git lfs track "*.joblib"`

---

## 🐳 Phương án 2: Docker + VPS/Cloud

### Tạo Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY environment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ project
COPY . .

# Expose port
EXPOSE 8501

# Chạy Streamlit
CMD ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build và chạy
```bash
# Build image
docker build -t streamlit-app .

# Chạy container
docker run -d -p 8501:8501 --name streamlit-app streamlit-app
```

### Deploy lên VPS
1. Upload code lên VPS (SSH, Git, hoặc SCP)
2. Cài Docker trên VPS
3. Build và chạy như trên
4. Cấu hình Nginx reverse proxy (tùy chọn)

---

## ☁️ Phương án 3: Heroku

### Tạo các file cần thiết

**Procfile:**
```
web: streamlit run src/app/app.py --server.port=$PORT --server.address=0.0.0.0
```

**runtime.txt:**
```
python-3.11.5
```

### Deploy
```bash
# Cài Heroku CLI
# Đăng nhập
heroku login

# Tạo app
heroku create your-app-name

# Deploy
git push heroku main

# Mở app
heroku open
```

### Lưu ý:
- ⚠️ Heroku free tier đã ngừng (cần trả phí)
- Cần thêm `Procfile` và `runtime.txt`

---

## 🌐 Phương án 4: AWS/Azure/GCP

### AWS (EC2 hoặc Elastic Beanstalk)
1. Tạo EC2 instance
2. SSH vào server
3. Cài Python, pip, Streamlit
4. Clone repo và chạy app
5. Cấu hình Security Group mở port 8501

### Azure App Service
1. Tạo App Service
2. Deploy từ GitHub hoặc Azure CLI
3. Cấu hình startup command: `streamlit run src/app/app.py`

### Google Cloud Run
1. Tạo Dockerfile (như Phương án 2)
2. Build và push lên Google Container Registry
3. Deploy lên Cloud Run

---

## 📋 Checklist trước khi deploy

- [ ] Đảm bảo models đã được train và lưu tại `src/models/`
- [ ] **QUAN TRỌNG**: Kiểm tra `.gitignore` - nếu models bị ignore, cần uncomment để commit models:
  ```bash
  # Sửa .gitignore, comment các dòng:
  # src/models/*.joblib
  # src/models/*_metadata.json
  ```
  Hoặc force add models:
  ```bash
  git add -f src/models/*.joblib src/models/*_metadata.json
  ```
- [ ] Kiểm tra `requirements.txt` có đầy đủ dependencies
- [ ] Test app chạy local: `streamlit run src/app/app.py`
- [ ] Đảm bảo đường dẫn file trong code đúng (relative paths)

---

## 🔧 Xử lý lỗi thường gặp

### Lỗi: "Module not found"
- Kiểm tra `requirements.txt` có đủ packages
- Đảm bảo import paths đúng

### Lỗi: "Model file not found"
- Kiểm tra models có trong repo
- Kiểm tra đường dẫn trong `app.py` (dùng relative paths)

### Lỗi: "Port already in use"
- Thay đổi port: `streamlit run src/app/app.py --server.port=8502`

---

## 💡 Tips

1. **Tối ưu file size**: Models `.joblib` có thể lớn, cân nhắc dùng Git LFS
2. **Environment variables**: Dùng `.streamlit/secrets.toml` cho thông tin nhạy cảm
3. **Caching**: App đã dùng `@st.cache_resource` để cache models
4. **Monitoring**: Có thể tích hợp logging để theo dõi

---

## 📞 Hỗ trợ

Nếu gặp vấn đề, kiểm tra:
- Logs của platform deploy
- Streamlit docs: https://docs.streamlit.io/
- GitHub Issues của project

