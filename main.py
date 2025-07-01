import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pandas as pd

# Bảng ánh xạ nội dung
content_mapping = {
    'do_quiz_ch1_theory_easy': {'id': 'quiz_001', 'chapter': 1, 'lesson': 'lesson_1.1', 'subcategory': 'theory', 'difficulty': 'easy', 'description': 'Quiz lý thuyết chương 1: Cơ bản về AI'},
    'read_pdf_ch1_theory_medium': {'id': 'pdf_001', 'chapter': 1, 'lesson': 'lesson_1.1', 'subcategory': 'theory', 'difficulty': 'medium', 'description': 'PDF: Khái niệm AI'},
    'watch_video_ch1_theory_easy': {'id': 'video_001', 'chapter': 1, 'lesson': 'lesson_1.1', 'subcategory': 'theory', 'difficulty': 'easy', 'description': 'Video: Giới thiệu về AI'}
}

# Bước 1: Chuẩn bị dữ liệu
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

# Bước 2: Phân nhóm học viên
def cluster_students(X):
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_ids = kmeans.fit_predict(X)
    return cluster_ids

# Bước 3: Dự đoán điểm số
def train_lr_model(X, y, cluster_ids, cluster_id):
    mask = cluster_ids == cluster_id
    lr = LinearRegression()
    lr.fit(X[mask], y[mask])
    return lr

# Bước 4: Chọn hành động tốt nhất
def choose_best_action(student_state, lr_model, actions):
    best_action = None
    best_score = -1
    for action in actions:
        next_state = student_state.copy()
        if action.startswith('do_quiz'):
            subcategory = action.split('_')[-2]
            next_state[1 if subcategory == 'theory' else 2] += 1
        elif action.startswith('read_pdf'):
            next_state[3] += 1
        elif action.startswith('watch_video'):
            next_state[0] += 1
        score_pred = lr_model.predict([next_state])[0]
        if score_pred > best_score:
            best_score = score_pred
            best_action = action
    return best_action, best_score

# Bước 5: Ánh xạ hành động
def map_action_to_content(action):
    return content_mapping.get(action, {'id': 'unknown', 'description': 'Unknown content'})

# Chạy hệ thống
if __name__ == "__main__":
    # Chuẩn bị dữ liệu
    X, y = prepare_data()
    
    # Phân nhóm học viên
    cluster_ids = cluster_students(X)
    print("Nhóm học viên:", cluster_ids)
    
    # Huấn luyện mô hình cho nhóm 0
    lr_model = train_lr_model(X, y, cluster_ids, cluster_id=0)
    
    # Trạng thái học viên: [video_views, quiz_done_theory, quiz_done_application, pdf_views, chapter]
    student_state = [2, 1, 0, 1, 1]
    actions = ['do_quiz_ch1_theory_easy', 'read_pdf_ch1_theory_medium', 'watch_video_ch1_theory_easy']
    
    # Chọn hành động
    best_action, score_pred = choose_best_action(student_state, lr_model, actions)
    content = map_action_to_content(best_action)
    
    print(f"Đề xuất: {content['description']} (ID: {content['id']}, Độ khó: {content['difficulty']})")
    print(f"Điểm dự đoán: {score_pred:.2f}")