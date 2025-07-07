import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dữ liệu mẫu với motivation
def prepare_data():
    data = pd.DataFrame({
        'video_views': [2, 1, 3, 0, 2],
        'quiz_done_theory': [2, 1, 3, 0, 2],
        'quiz_done_application': [1, 1, 1, 1, 1],
        'pdf_views': [1, 2, 0, 3, 1],
        'chapter': [1, 1, 1, 1, 1],
        'motivation': [4.0, 3.0, 4.5, 2.0, 3.8]
    })
    X = data[['video_views', 'quiz_done_theory', 'quiz_done_application', 'pdf_views', 'chapter']].values
    y = data['motivation'].values
    return X, y

# Hồi quy tuyến tính dự đoán động lực
def linear_regression_predict_motivation():
    X, y = prepare_data()
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Dự đoán động lực
    y_pred = model.predict(X_test)
    
    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)
    print("Hồi quy tuyến tính dự đoán động lực")
    print(f"Hệ số: {model.coef_}")
    print(f"Hằng số chặn: {model.intercept_}")
    print(f"Mean Squared Error: {mse}")
    print(f"Dự đoán động lực cho X_test: {y_pred}")
    
    # Ví dụ dự đoán cho học viên mới
    new_student = np.array([[3, 2, 1, 0, 1]])  # 3 video, 2 quiz lý thuyết, 1 quiz ứng dụng, 0 PDF, chương 1
    predicted_motivation = model.predict(new_student)
    print(f"Dự đoán động lực cho học viên mới: {predicted_motivation[0]}")
    
    return model, predicted_motivation[0]

# Tích hợp động lực vào Q-table
def integrate_motivation_to_qlearning(predicted_motivation):
    # Giả sử Q-table với 10 trạng thái, 5 hành động
    q_table = np.zeros((10, 5))
    state_idx = 0  # Trạng thái giả định
    action_idx = 0  # Hành động: ví dụ "làm quiz nâng cao"
    
    # Sử dụng động lực làm phần thưởng
    reward = predicted_motivation
    learning_rate = 0.1
    discount_factor = 0.9
    
    # Cập nhật Q-table
    q_table[state_idx, action_idx] = (1 - learning_rate) * q_table[state_idx, action_idx] + \
                                     learning_rate * (reward + discount_factor * np.max(q_table[state_idx]))
    
    print(f"Q-table sau khi cập nhật với động lực {reward}: {q_table[state_idx, action_idx]}")
    return q_table

# Chạy chương trình
if __name__ == "__main__":
    print("--- Dự đoán động lực ---")
    model, predicted_motivation = linear_regression_predict_motivation()
    print("\n--- Tích hợp vào Q-learning ---")
    q_table = integrate_motivation_to_qlearning(predicted_motivation)