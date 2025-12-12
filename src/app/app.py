"""
Demo App - Dự đoán Điểm Học tập Học sinh
Streamlit app để dự đoán điểm cuối kỳ (G3) dựa trên thông tin học sinh
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

# Thêm parent directory vào path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.predict_model import load_model_and_preprocessor, load_metadata


@st.cache_resource
def load_models():
    """Load models và preprocessor (cache để tăng tốc)"""
    base_path = Path(__file__).parent.parent.parent / "src" / "models"
    
    models = {}
    preprocessor = None
    
    try:
        # Load preprocessor
        preprocessor_path = base_path / "preprocessor.joblib"
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
        
        # Load Model A
        model_a_path = base_path / "model_a_regression.joblib"
        if model_a_path.exists():
            models['Model A (Random Forest)'] = {
                'model': joblib.load(model_a_path),
                'metadata': load_metadata(str(model_a_path))
            }
        
        # Load Model B
        model_b_path = base_path / "model_b_regression.joblib"
        if model_b_path.exists():
            models['Model B (SVM)'] = {
                'model': joblib.load(model_b_path),
                'metadata': load_metadata(str(model_b_path))
            }
    
    except Exception as e:
        st.error(f"Lỗi khi load models: {e}")
    
    return models, preprocessor


def create_input_form():
    """Tạo form nhập liệu"""
    st.header("📝 Thông tin Học sinh")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Thông tin Cá nhân")
        school = st.selectbox(
            "Trường", 
            ["GP", "MS"],
            format_func=lambda x: "Gabriel Pereira (GP)" if x == "GP" else "Mousinho da Silveira (MS)",
            help="GP: Gabriel Pereira, MS: Mousinho da Silveira - Hai trường trung học ở Bồ Đào Nha"
        )
        sex = st.selectbox("Giới tính", ["F", "M"], help="F: Nữ, M: Nam")
        age = st.number_input("Tuổi", min_value=15, max_value=22, value=17)
        address = st.selectbox("Địa chỉ", ["U", "R"], help="U: Thành thị, R: Nông thôn")
        famsize = st.selectbox("Quy mô gia đình", ["LE3", "GT3"], help="LE3: ≤3 người, GT3: >3 người")
        Pstatus = st.selectbox("Tình trạng cha mẹ", ["T", "A"], help="T: Sống cùng, A: Ly thân")
    
    with col2:
        st.subheader("Giáo dục Gia đình")
        Medu = st.selectbox("Trình độ học vấn Mẹ", 
                           [0, 1, 2, 3, 4],
                           format_func=lambda x: {
                               0: "Không có",
                               1: "Tiểu học (lớp 4)",
                               2: "Lớp 5-9",
                               3: "Trung học",
                               4: "Đại học"
                           }[x])
        Fedu = st.selectbox("Trình độ học vấn Cha",
                           [0, 1, 2, 3, 4],
                           format_func=lambda x: {
                               0: "Không có",
                               1: "Tiểu học (lớp 4)",
                               2: "Lớp 5-9",
                               3: "Trung học",
                               4: "Đại học"
                           }[x])
        Mjob = st.selectbox("Nghề nghiệp Mẹ", 
                          ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Nghề nghiệp Cha",
                          ["teacher", "health", "services", "at_home", "other"])
        reason = st.selectbox("Lý do chọn trường",
                            ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Người giám hộ",
                              ["mother", "father", "other"])
    
    st.subheader("Học tập & Hoạt động")
    col3, col4 = st.columns(2)
    
    with col3:
        traveltime = st.selectbox("Thời gian đi học",
                                 [1, 2, 3, 4],
                                 format_func=lambda x: {
                                     1: "<15 phút",
                                     2: "15-30 phút",
                                     3: "30 phút - 1 giờ",
                                     4: ">1 giờ"
                                 }[x])
        studytime = st.selectbox("Thời gian học/tuần",
                                [1, 2, 3, 4],
                                format_func=lambda x: {
                                    1: "<2 giờ",
                                    2: "2-5 giờ",
                                    3: "5-10 giờ",
                                    4: ">10 giờ"
                                }[x])
        failures = st.number_input("Số lần trượt trước đây", min_value=0, max_value=4, value=0)
        schoolsup = st.selectbox("Hỗ trợ từ trường", ["yes", "no"])
        famsup = st.selectbox("Hỗ trợ từ gia đình", ["yes", "no"])
        paid = st.selectbox("Lớp học thêm có trả phí", ["yes", "no"])
    
    with col4:
        activities = st.selectbox("Hoạt động ngoại khóa", ["yes", "no"])
        nursery = st.selectbox("Đã học mẫu giáo", ["yes", "no"])
        higher = st.selectbox("Muốn học đại học", ["yes", "no"])
        internet = st.selectbox("Có internet ở nhà", ["yes", "no"])
        romantic = st.selectbox("Có người yêu", ["yes", "no"])
    
    st.subheader("Sức khỏe & Xã hội")
    col5, col6 = st.columns(2)
    
    with col5:
        famrel = st.slider("Chất lượng quan hệ gia đình", 1, 5, 4,
                          help="1: Rất tệ, 5: Rất tốt")
        freetime = st.slider("Thời gian rảnh", 1, 5, 3,
                           help="1: Rất ít, 5: Rất nhiều")
        goout = st.slider("Đi chơi với bạn", 1, 5, 3,
                        help="1: Rất ít, 5: Rất nhiều")
    
    with col6:
        Dalc = st.slider("Uống rượu ngày thường", 1, 5, 1,
                       help="1: Rất ít, 5: Rất nhiều")
        Walc = st.slider("Uống rượu cuối tuần", 1, 5, 1,
                       help="1: Rất ít, 5: Rất nhiều")
        health = st.slider("Tình trạng sức khỏe", 1, 5, 3,
                         help="1: Rất tệ, 5: Rất tốt")
        absences = st.number_input("Số ngày nghỉ học", min_value=0, max_value=93, value=0)
    
    # Tạo DataFrame từ input
    data = {
        'school': [school],
        'sex': [sex],
        'age': [age],
        'address': [address],
        'famsize': [famsize],
        'Pstatus': [Pstatus],
        'Medu': [Medu],
        'Fedu': [Fedu],
        'Mjob': [Mjob],
        'Fjob': [Fjob],
        'reason': [reason],
        'guardian': [guardian],
        'traveltime': [traveltime],
        'studytime': [studytime],
        'failures': [failures],
        'schoolsup': [schoolsup],
        'famsup': [famsup],
        'paid': [paid],
        'activities': [activities],
        'nursery': [nursery],
        'higher': [higher],
        'internet': [internet],
        'romantic': [romantic],
        'famrel': [famrel],
        'freetime': [freetime],
        'goout': [goout],
        'Dalc': [Dalc],
        'Walc': [Walc],
        'health': [health],
        'absences': [absences]
    }
    
    return pd.DataFrame(data)


def convert_to_gpa(g3_score):
    """
    Chuyển đổi điểm G3 (0-20) sang GPA và Điểm
    
    Parameters:
    -----------
    g3_score : float
        Điểm G3 (0-20)
    
    Returns:
    --------
    dict: {'G3': float, 'GPA_4.0': float, 'GPA_10': float}
        - G3: Điểm gốc (0-20)
        - GPA_4.0: Grade Point Average thang 4.0 (0-4)
        - GPA_10: Điểm thang 10 (0-10) - không phải GPA
    """
    return {
        'G3': round(g3_score, 2),
        'GPA_4.0': round((g3_score / 20) * 4, 2),
        'GPA_10': round((g3_score / 20) * 10, 2)
    }


def predict_score(model, preprocessor, X):
    """Dự đoán điểm"""
    try:
        if preprocessor is not None:
            X_processed = preprocessor.transform(X)
        else:
            X_processed = X
        
        prediction_raw = model.predict(X_processed)[0]
        
        # Debug: hiển thị prediction raw (chỉ trong development)
        if st.session_state.get('debug_mode', False):
            st.write(f"🔍 Debug - Prediction raw: {prediction_raw:.4f}")
            st.write(f"🔍 Debug - X shape: {X.shape}, X_processed shape: {X_processed.shape}")
            st.write(f"🔍 Debug - X columns: {list(X.columns)}")
        
        prediction = max(0, min(20, round(prediction_raw, 2)))  # Giới hạn trong [0, 20]
        return prediction
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        import traceback
        st.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        return None


def main():
    st.set_page_config(
        page_title="Dự đoán Điểm Học tập",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Dự đoán Điểm Học tập Học sinh")
    st.markdown("""
    Ứng dụng sử dụng Machine Learning để dự đoán điểm cuối kỳ (G3) của học sinh 
    dựa trên thông tin cá nhân, gia đình, học tập và xã hội.
    
    **Dataset**: Student Performance (UCI ML Repository)
    """)
    
    # Load models
    models, preprocessor = load_models()
    
    if not models:
        st.error("❌ Không tìm thấy models. Vui lòng train models trước.")
        st.stop()
    
    if preprocessor is None:
        st.warning("⚠️ Không tìm thấy preprocessor. Predictions có thể không chính xác.")
    
    # Sidebar
    st.sidebar.header("⚙️ Cài đặt")
    selected_model = st.sidebar.selectbox(
        "Chọn Model",
        list(models.keys())
    )
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("🔍 Debug Mode", value=False, help="Hiển thị thông tin debug khi predict")
    st.session_state['debug_mode'] = debug_mode
    
    # Hiển thị thông tin model
    if models[selected_model]['metadata']:
        metadata = models[selected_model]['metadata']
        st.sidebar.markdown("### Thông tin Model")
        st.sidebar.write(f"**Loại**: {metadata.get('model_type', 'N/A')}")
        st.sidebar.write(f"**Thuật toán**: {metadata.get('model_name', 'N/A')}")
        if 'best_params' in metadata:
            st.sidebar.write("**Hyperparameters:**")
            for key, value in metadata['best_params'].items():
                st.sidebar.write(f"  - {key}: {value}")
        
        # Hiển thị R2 score nếu có
        if 'training_history' in metadata and 'best_score' in metadata['training_history']:
            r2_score = metadata['training_history']['best_score']
            st.sidebar.write(f"**R² Score**: {r2_score:.4f}")
            if r2_score < 0.3:
                st.sidebar.warning("⚠️ Model performance thấp (R² < 0.3). Predictions có thể không chính xác.")
    
    # Form nhập liệu
    X = create_input_form()
    
    # Predict button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_button = st.button("🔮 Dự đoán Điểm", type="primary", use_container_width=True)
    
    # Kết quả
    if predict_button:
        with st.spinner("Đang dự đoán..."):
            model = models[selected_model]['model']
            prediction = predict_score(model, preprocessor, X)
        
        if prediction is not None:
            st.markdown("---")
            st.header("📈 Kết quả Dự đoán")
            
            # Chuyển đổi sang GPA
            gpa_scores = convert_to_gpa(prediction)
            
            # Hiển thị điểm G3
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            with col_res2:
                st.metric(
                    label="Điểm cuối kỳ dự đoán (G3)",
                    value=f"{gpa_scores['G3']:.2f}",
                    help="Thang điểm 0-20 (Bồ Đào Nha)"
                )
            
            # Hiển thị GPA và Điểm
            st.subheader("📊 Chuyển đổi sang GPA và Điểm")
            col_gpa1, col_gpa2 = st.columns(2)
            with col_gpa1:
                st.metric(
                    label="GPA (thang 4.0)",
                    value=f"{gpa_scores['GPA_4.0']:.2f}",
                    help="Grade Point Average - Thang điểm 0-4 (hệ thống Mỹ)"
                )
            with col_gpa2:
                st.metric(
                    label="Điểm (thang 10)",
                    value=f"{gpa_scores['GPA_10']:.2f}",
                    help="Thang điểm 0-10 (phổ biến ở Việt Nam)"
                )
            
            # Đánh giá dựa trên thang điểm 10 (phổ biến ở VN)
            diem_10 = gpa_scores['GPA_10']
            if diem_10 >= 8.5:
                st.success("🎉 Xuất sắc! Học sinh có khả năng đạt điểm cao (Điểm ≥ 8.5).")
            elif diem_10 >= 7.0:
                st.info("👍 Tốt! Học sinh có khả năng đạt điểm khá (7.0 ≤ Điểm < 8.5).")
            elif diem_10 >= 5.0:
                st.warning("⚠️ Trung bình. Cần cải thiện thêm (5.0 ≤ Điểm < 7.0).")
            else:
                st.error("❌ Yếu. Cần hỗ trợ và cải thiện nhiều (Điểm < 5.0).")
            
            # Hiển thị thông tin đã nhập
            with st.expander("📋 Xem lại thông tin đã nhập"):
                st.dataframe(X, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>Dataset: Student Performance (UCI ML Repository) | 
        Đồ án Khai thác Dữ liệu</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

