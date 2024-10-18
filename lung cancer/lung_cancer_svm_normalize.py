import numpy as np

# Hàm đọc dữ liệu từ file
def get_data(filename):
    dataf = open(filename, encoding='utf8').readlines()
    datai = []
    for i in dataf:
        int_array = [int(x) for x in i.split(',')]
        datai.append(int_array)
    return datai

# Hàm chuẩn hóa dữ liệu Min-Max scaling
def normalize_min_max(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized

# Bước 1: Chuẩn bị dữ liệu
datai = get_data('lung_cancer_traning.csv')
data = np.array(datai)

X = data[:, :-1]  # Tách các thuộc tính (features)
y = data[:, -1]   # Tách nhãn (labels)

# Chuẩn hóa dữ liệu đầu vào
X = normalize_min_max(X)

# Chuẩn hóa đầu ra nhãn: SVM yêu cầu các nhãn phải là -1 và 1
y = np.where(y == 2, 1, -1)

# Bước 2: Khởi tạo trọng số và bias
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.001
epochs = 1000  # Số vòng lặp

# Bước 3: Hàm huấn luyện mô hình SVM
def train_svm(X, y, weights, bias, learning_rate, epochs):
    n_samples, n_features = X.shape
    for epoch in range(epochs):
        for idx, x_i in enumerate(X):
            condition = y[idx] * (np.dot(x_i, weights) + bias)
            if condition >= 1:
                # Gradient Descent cho trường hợp chính xác
                weights -= learning_rate * (2 * 1/epochs * weights)
            else:
                # Gradient Descent cho trường hợp sai
                weights -= learning_rate * (2 * 1/epochs * weights - np.dot(x_i, y[idx]))
                bias -= learning_rate * y[idx]
    return weights, bias

# Bước 4: Huấn luyện mô hình
weights, bias = train_svm(X, y, weights, bias, learning_rate, epochs)

# Bước 5: Dự đoán
def predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return np.sign(linear_output)

# Dữ liệu huấn luyện
datai = get_data('lung_cancer_testing.csv')
data = np.array(datai)

X = data[:, :-1]
y = data[:, -1]  # Tách nhãn (labels)

# Chuẩn hóa dữ liệu đầu vào
X = normalize_min_max(X)

# Chuẩn hóa đầu ra nhãn: SVM yêu cầu các nhãn phải là -1 và 1
y = np.where(y == 2, 1, -1)

# Kiểm tra dự đoán trên tập dữ liệu
predictions = predict(X, weights, bias)

# Tính độ chính xác
dung = 0
sai = 0
for idy, y_i in enumerate(y):
    if y_i == predictions[idy]:
        dung += 1
    else: 
        sai += 1

do_chinh_xac = (dung / len(y)) * 100
print(f"Độ chính xác của mô hình: {do_chinh_xac:.2f}%")

print(f"Trọng số: {weights}")
print(f"Bias: {bias}")
