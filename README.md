# 🎓 Demo AI cho Khóa Luận Tốt Nghiệp

## 🎯 Mục tiêu
Ứng dụng AI để cá nhân hóa học tập bằng cách:
- Phân nhóm học sinh (`KMeans`)
- Dự đoán điểm số (`Linear Regression`)
- Gợi ý nội dung học tập phù hợp (`Q-Learning`)

---

## ⚙️ Thành phần

### 🔹 Tiền xử lý dữ liệu
Chuẩn hóa dữ liệu hành vi học sinh bao gồm:
- `video_views`: Số lượt xem video
- `quiz_done_theory`: Số lần làm quiz lý thuyết
- `quiz_done_application`: Số lần làm quiz ứng dụng
- `pdf_views`: Số lượt đọc tài liệu PDF
- `chapter`: Chương học tương ứng

> Áp dụng chuẩn hóa Z-score để đảm bảo dữ liệu cùng thang đo.

---

### 🔹 Phân cụm bằng KMeans
- Sử dụng `KMeans(n_clusters=3)` từ thư viện `scikit-learn`
- Mục đích: phân học sinh thành 3 nhóm có hành vi học tập tương đồng

---

### 🔹 Dự đoán điểm bằng Linear Regression
- Huấn luyện một mô hình hồi quy tuyến tính cho mỗi cụm
- Dự đoán `quiz_score` dựa trên hành vi học tập
- Giúp đánh giá ảnh hưởng của từng hành động đến kết quả học tập

---

### 🔹 Q-Learning (Học tăng cường)
- Ánh xạ trạng thái học tập → hành động tối ưu
- Hành động gồm: làm quiz lý thuyết, làm quiz ứng dụng, đọc PDF,...
- Phần thưởng (`reward`) là điểm số dự đoán sau hành động
- Bảng Q-table được cập nhật liên tục để cải thiện gợi ý

---

### 🔹 Gợi ý nội dung học tập
- Q-learning sẽ chọn hành động có Q-value cao nhất
- Hành động đó được ánh xạ đến nội dung cụ thể từ `content_mapping`
- Ví dụ: `'do_quiz_ch2_theory'` → Quiz lý thuyết chương 2

---

## 🛠 Công nghệ sử dụng
- Python 3.x
- `pandas`, `numpy` để xử lý dữ liệu
- `scikit-learn` cho KMeans và Linear Regression
- Q-Learning tự cài đặt bằng Python thuần

---

## ✅ Kết quả đầu ra
- Phân cụm học sinh
- Dự đoán điểm số sau mỗi hành động học tập
- Gợi ý hành động tốt nhất để cải thiện điểm số
- Gợi ý nội dung học phù hợp theo từng học sinh

---

## 📚 Ứng dụng thực tế
- Có thể áp dụng trong các hệ thống LMS để gợi ý học tập cá nhân hóa
- Góp phần phát triển mô hình học tập thích ứng trong giáo dục STEM
