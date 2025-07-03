import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path

# ===================== CONFIG =====================
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


def prepare_data_smart(n_samples: int = 1_000, random_state: int | None = None):
    """
    Sinh dữ liệu hành vi học tập mô phỏng có *tương quan logic*:
    - Học viên chăm (engagement cao) => xem video & làm quiz nhiều, điểm cao hơn.
    - Học viên ít làm bài thì điểm thấp hơn.
    """
    rng = np.random.default_rng(random_state)

    # 1) Tạo thang đo "engagement" 0‒1 theo phân bố Beta (skew về phía ít chăm)
    engagement = rng.beta(a=2, b=5, size=n_samples)

    # 2) Sinh các đặc trưng rời rạc chịu ảnh hưởng của engagement
    #    (dùng phân bố Poisson rồi cắt ngưỡng để giữ trong biên)
    video_views = np.clip(rng.poisson(lam=engagement * 10), 0, 15)
    quiz_done_theory = np.clip(rng.poisson(lam=engagement * 8), 0, 10)

    #    Bài áp dụng thường ít hơn bài lý thuyết
    quiz_done_application = np.clip(
        rng.binomial(n=quiz_done_theory, p=0.5), 0, None
    )

    #    PDF thường được mở kèm khi xem video
    pdf_views = np.clip(
        rng.poisson(lam=video_views * rng.uniform(0.2, 0.6, n_samples)), 0, 10
    )

    # 3) Chương học (1‒6) – càng engagement cao càng có xu hướng ở chương cao
    chapter = rng.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.25, 0.20, 0.18, 0.15, 0.12, 0.10])
    chapter = np.clip(chapter + (engagement * 2).astype(int), 1, 6)

    # 4) Tính điểm: 60% phụ thuộc *tỷ lệ đúng* (số bài làm) + 20% xem video + 20% noise
    noise = rng.normal(0, 1.2, n_samples)
    quiz_score_raw = (
        0.6 * (quiz_done_theory + quiz_done_application)
        + 0.2 * video_views
        + noise
    )
    quiz_score = np.round(np.clip(quiz_score_raw / 2, 0, 10)).astype(int)

    df = pd.DataFrame(
        {
            "video_views": video_views,
            "quiz_done_theory": quiz_done_theory,
            "quiz_done_application": quiz_done_application,
            "pdf_views": pdf_views,
            "chapter": chapter,
            "quiz_score": quiz_score,
        }
    )

    X = df.drop(columns="quiz_score").values
    return df, X

# ============ 1. CHUẨN BỊ DỮ LIỆU ================
def prepare_data():
    df = pd.DataFrame({
        'video_views':           [2, 1, 3, 0, 2],
        'quiz_done_theory':      [2, 1, 3, 0, 2],
        'quiz_done_application': [1, 1, 1, 1, 1],
        'pdf_views':             [1, 2, 0, 3, 1],
        'chapter':               [1, 1, 1, 1, 1],
        'quiz_score':            [4, 3, 5, 2, 4],
    })
    X = df[['video_views', 'quiz_done_theory',
            'quiz_done_application', 'pdf_views', 'chapter']].values
    return df, X

# ============ 2. KMEANS + LƯU CSV ================
def cluster_and_save(df, X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster_id"] = kmeans.fit_predict(X)
    
    csv_path = OUTPUT_DIR / "student_clusters.csv"
    df.to_csv(csv_path, index=False)
    print(f"✔ Đã ghi file: {csv_path}")

    return df, kmeans.labels_

# ============ 3A. VẼ BIỂU ĐỒ CỘT ================
def plot_feature_bar(df):
    cols = ['video_views', 'quiz_done_theory',
            'quiz_done_application', 'pdf_views']
    ax = df[cols].plot(kind="bar", figsize=(10,5))
    ax.set_title("Feature counts for each student")
    ax.set_xlabel("Student index")
    ax.set_ylabel("Count")
    ax.legend(loc='upper right')
    
    img_path = OUTPUT_DIR / "feature_bar.png"
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"✔ Đã lưu hình: {img_path}")

# ============ 3B. SCATTER PCA ================

# ============ 3B-alt. SCATTER PCA 3-D ================
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (kích hoạt 3D)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_pca_scatter_3d(X, labels):
    """
    Vẽ scatter 3-D của các cụm KMeans sau khi giảm chiều bằng PCA.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    labels : ndarray (n_samples,)
        Nhãn cụm KMeans.
    """
    pca = PCA(n_components=3, random_state=0)
    X_3d = pca.fit_transform(X)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    for cid in np.unique(labels):
        mask = labels == cid
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            label=f"Cluster {cid}",
            s=40,
            depthshade=True,
        )

    ax.set_title("KMeans clusters (PCA 3-D projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    img_path = OUTPUT_DIR / "cluster_scatter_3d.png"
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"✔ Đã lưu hình: {img_path}")

def plot_pca_scatter(X, labels):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(8,5))
    for cid in np.unique(labels):
        mask = labels == cid
        plt.scatter(X_2d[mask,0], X_2d[mask,1],
                    label=f"Cluster {cid}", s=80)
    plt.title("KMeans clusters (PCA 2-D projection)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend()
    
    img_path = OUTPUT_DIR / "cluster_scatter.png"
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"✔ Đã lưu hình: {img_path}")
    
import pandas as pd
import seaborn as sns

def analyze_cluster_characteristics(df, cluster_ids):
    df['cluster_id'] = cluster_ids
    cluster_summary = df.groupby('cluster_id').mean()
    print("Đặc trưng trung bình của mỗi cụm:")
    print(cluster_summary)
    return cluster_summary

def plot_correlation_matrix(df, cluster_id):
    cluster_data = df[df['cluster_id'] == cluster_id]
    corr_matrix = cluster_data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(f"Ma trận tương quan - Cụm {cluster_id}")
    plt.savefig(f"outputs/corr_matrix_cluster_{cluster_id}.png", dpi=150)
    plt.close()
    print(f"✔ Đã lưu hình: outputs/corr_matrix_cluster_{cluster_id}.png")



# ================== MAIN =========================
if __name__ == "__main__":
    # df, X = prepare_data()
    df, X = prepare_data_smart(n_samples=5_000, random_state=42)
    
    df, labels = cluster_and_save(df, X, n_clusters=3)
    
    plot_feature_bar(df)          # Hình 1
    plot_pca_scatter(X, labels)   # Hình 2
    plot_pca_scatter_3d(X, labels)
    cluster_summary = analyze_cluster_characteristics(df, labels)
    for cid in df['cluster_id'].unique():
        plot_correlation_matrix(df, cid)