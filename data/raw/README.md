# Student Performance Dataset

## Nguồn
UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/320/student+performance

## Mô tả
Dataset dự đoán kết quả học tập của học sinh trung học (high school).

## Thông tin Dataset
- **Số dòng**: 649
- **Số cột**: 33 (30 features + 3 target variables: G1, G2, G3)
- **Missing values**: Không có
- **Loại bài toán**: Regression (dự đoán điểm G3 - final grade)

## Biến mục tiêu
- **G3**: Điểm cuối kỳ (final grade) - từ 0 đến 20
- **G1**: Điểm kỳ 1 (first period grade) - từ 0 đến 20
- **G2**: Điểm kỳ 2 (second period grade) - từ 0 đến 20

**Lưu ý**: G3 có tương quan mạnh với G1 và G2. Để dự đoán hữu ích hơn, có thể loại bỏ G1 và G2 khỏi features khi training.

## Features
- **Demographic**: 
  - school: Trường học (GP: Gabriel Pereira, MS: Mousinho da Silveira)
  - sex, age, address
- **Family**: famsize, Pstatus, Medu, Fedu, Mjob, Fjob, guardian, famsup
- **Academic**: traveltime, studytime, failures, schoolsup, paid, activities, nursery, higher
- **Social**: internet, romantic, famrel, freetime, goout
- **Health**: Dalc, Walc, health, absences

## Citation
Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

