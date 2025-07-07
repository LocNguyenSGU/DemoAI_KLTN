import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dữ liệu mẫu
def prepare_data():
    data = pd.DataFrame({
        'video_views': [2, 1, 3, 0, 2],
        'quiz_done_theory': [2, 1, 3, 0, 2],
        'quiz_done_application': [1, 1, 1, 1, 1],
        'pdf_views': [1, 2, 0, 3, 1],
        'chapter': [1, 1, 1, 1, 1],
        'quiz_score': [4, 3, 5, 2, 4]
    })
    X = data[['video_views', 'quiz_done_theory', 'quiz_done_application', 'pdf_views', 'chapter']].values
    y = data['quiz_score'].values
    return X, y

# Hướng 1: Dự đoán điểm số độc lập
def linear_regression_predict_score():
    X, y = prepare_data()
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Huấn luyện mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Dự đoán điểm số
    y_pred = model.predict(X_test)
    
    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)
    print("Hướng 1: Dự đoán điểm số độc lập")
    print(f"Hệ số: {model.coef_}")
    print(f"Hằng số chặn: {model.intercept_}")
    print(f"Mean Squared Error: {mse}")
    print(f"Dự đoán điểm số cho X_test: {y_pred}")
    
    # Ví dụ dự đoán cho học viên mới
    new_student = np.array([[3, 2, 1, 0, 1]])  # 3 video, 2 quiz lý thuyết, 1 quiz ứng dụng, 0 PDF, chương 1
    predicted_score = model.predict(new_student)
    print(f"Dự đoán điểm số cho học viên mới: {predicted_score[0]}")

# Hướng 2: Hỗ trợ Q-learning bằng cách ước lượng phần thưởng
def linear_regression_for_qlearning():
    X, y = prepare_data()
    
    # Huấn luyện mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X, y)
    
    # Dự đoán điểm số làm phần thưởng cho Q-learning
    # Giả sử trạng thái hiện tại và hành động
    state_action = np.array([[3, 2, 1, 0, 1]])  # Ví dụ: học viên xem 3 video, 2 quiz lý thuyết, 1 quiz ứng dụng
    predicted_reward = model.predict(state_action)[0]
    
    print("Hướng 2: Ước lượng phần thưởng cho Q-learning")
    print(f"Hệ số: {model.coef_}")
    print(f"Hằng số chặn: {model.intercept_}")
    print(f"Phần thưởng dự đoán (dựa trên điểm số): {predicted_reward}")
    
    # Ví dụ tích hợp vào Q-learning
    # Giả sử Q-table là một dictionary hoặc numpy array
    q_table = np.zeros((10, 5))  # Ví dụ: 10 trạng thái, 5 hành động
    state_idx = 0  # Chỉ số trạng thái (giả định)
    action_idx = 0  # Hành động: ví dụ "làm quiz nâng cao"
    
    # Cập nhật Q-table với phần thưởng từ hồi quy tuyến tính
    learning_rate = 0.1
    discount_factor = 0.9
    q_table[state_idx, action_idx] = (1 - learning_rate) * q_table[state_idx, action_idx] + \
                                     learning_rate * (predicted_reward + discount_factor * np.max(q_table[state_idx]))
    
    print(f"Q-table sau khi cập nhật: {q_table[state_idx, action_idx]}")

# Chạy cả hai hướng
if __name__ == "__main__":
    print("--- Hướng 1 ---")
    linear_regression_predict_score()
    print("\n--- Hướng 2 ---")
    linear_regression_for_qlearning()