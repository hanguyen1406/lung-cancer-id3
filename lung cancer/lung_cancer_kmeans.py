import numpy as np
from collections import Counter

# Hàm tính khoảng cách Euclidean giữa hai điểm
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# Thuật toán K-Means
def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    
    # Bước 1: Khởi tạo ngẫu nhiên các centroid
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    
    for _ in range(max_iters):
        # Bước 2: Gán mỗi điểm vào cụm gần nhất
        clusters = [[] for _ in range(k)]
        for idx, sample in enumerate(X):
            distances = [euclidean_distance(sample, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(idx)

        # Bước 3: Lưu các centroid trước đó để so sánh
        previous_centroids = centroids.copy()

        # Bước 4: Cập nhật lại các centroid
        for cluster_idx, cluster in enumerate(clusters):
            if cluster:  # Kiểm tra nếu cụm không rỗng
                cluster_mean = np.mean(X[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean

        # Bước 5: Kiểm tra điều kiện dừng (nếu không thay đổi centroid)
        is_converged = np.all(previous_centroids == centroids)
        if is_converged:
            break
    
    # Trả về các centroid và các điểm trong cụm
    return centroids, clusters

# Hàm tính độ chính xác
def calculate_accuracy(clusters, y_true):
    cluster_labels = np.zeros(len(y_true))

    # Duyệt qua từng cụm
    for cluster_idx, cluster in enumerate(clusters):
        if not cluster:  # Kiểm tra nếu cụm rỗng
            continue

        # Lấy nhãn thực tế cho các điểm trong cụm
        true_labels_in_cluster = y_true[cluster]

        # Tìm nhãn phổ biến nhất trong cụm và gán cho toàn bộ cụm
        most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
        for idx in cluster:
            cluster_labels[idx] = most_common_label

    # Tính độ chính xác bằng cách so sánh nhãn dự đoán và nhãn thực tế
    accuracy = np.sum(cluster_labels == y_true) / len(y_true)
    return accuracy * 100

# Chuẩn bị dữ liệu
def get_data(filename):
    dataf = open(filename, encoding='utf8').readlines()
    datai = []
    for i in dataf:
        int_array = [int(x) for x in i.split(',')]
        datai.append(int_array)
    return datai

# Tải dữ liệu
datai = get_data('lung_cancer_traning.csv')
data = np.array(datai)

X = data[:, :-1]  # Chỉ sử dụng các thuộc tính, không sử dụng nhãn lung_cancer
y_true = data[:, -1]  # Nhãn thực tế

# Chạy thuật toán K-Means với 2 cụm
k = 2
centroids, clusters = kmeans(X, k)

# Tính độ chính xác của mô hình
accuracy = calculate_accuracy(clusters, y_true)
print(f"Độ chính xác của mô hình K-Means: {accuracy:.2f}%")
