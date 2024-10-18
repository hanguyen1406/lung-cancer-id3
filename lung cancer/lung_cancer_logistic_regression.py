import numpy as np

# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm tính Log Loss
def log_loss(y, y_hat):
    m = len(y)
    return -1/m * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

# Hàm chuẩn hóa dữ liệu
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Hàm huấn luyện Logistic Regression
def logistic_regression(X, y, learning_rate, num_iterations):
    m, n = X.shape  # m là số mẫu, n là số đặc trưng (features)
    
    # Khởi tạo các tham số beta với giá trị 0
    beta = np.zeros(n)
    bias = 0
    
    for i in range(num_iterations):
        # Dự đoán
        linear_model = np.dot(X, beta) + bias
        y_hat = sigmoid(linear_model)
        
        # Tính gradient
        dbeta = 1/m * np.dot(X.T, (y_hat - y))
        dbias = 1/m * np.sum(y_hat - y)
        
        # Cập nhật tham số
        beta -= learning_rate * dbeta
        bias -= learning_rate * dbias
        
        # Tính loss sau mỗi 1000 bước để theo dõi
        if i % 1000 == 0:
            loss = log_loss(y, y_hat)
            print(f"Iteration {i}, Loss: {loss}")
    
    return beta, bias

# Hàm dự đoán
def predict(X, beta, bias):
    linear_model = np.dot(X, beta) + bias
    y_hat = sigmoid(linear_model)
    return [1 if i > 0.5 else 0 for i in y_hat]

# Chuẩn bị dữ liệu
data = np.array([
    [0,69,1,2,2,1,1,2,1,2,2,2,2,2,2,1],
    [0,74,2,1,1,1,2,2,2,1,1,1,2,2,2,1],
    [1,59,1,1,1,2,1,2,1,2,1,2,2,1,2,2],
    [0,63,2,2,2,1,1,1,1,1,2,1,1,2,2,2],
    [1,63,1,2,1,1,1,1,1,2,1,2,2,1,1,2],
    [1,75,1,2,1,1,2,2,2,2,1,2,2,1,1,1],
    [0,52,2,1,1,1,1,2,1,2,2,2,2,1,2,1],
    [1,51,2,2,2,2,1,2,2,1,1,1,2,2,1,1],
    [1,68,2,1,2,1,1,2,1,1,1,1,1,1,1,2],
    [0,53,2,2,2,2,2,1,2,1,2,1,1,2,2,1],
    [1,61,2,2,2,2,2,2,1,2,1,2,2,2,1,1],
    [0,72,1,1,1,1,2,2,2,2,2,2,2,1,2,1],
    [1,60,2,1,1,1,1,2,1,1,1,1,2,1,1,2],
    [0,58,2,1,1,1,1,2,2,2,2,2,2,1,2,1],
    [0,69,2,1,1,1,1,1,2,2,2,2,1,1,2,2],
    [1,48,1,2,2,2,2,2,2,2,1,2,2,2,1,1],
    [0,75,2,1,1,1,2,1,2,2,2,2,2,1,2,1],
    [0,57,2,2,2,2,2,1,1,1,2,1,1,2,2,1],
    [1,68,2,2,2,2,2,2,1,1,1,2,2,1,1,1],
    [1,61,1,1,1,1,2,2,1,1,1,1,2,1,1,2],
    [1,44,2,2,2,2,2,2,1,1,1,1,2,2,1,1],
    [1,64,1,2,2,2,1,1,2,2,1,2,1,2,1,1]
])

# Tách đặc trưng (X) và nhãn (y)
X = data[:, :-1]  # Tất cả các cột trừ cột cuối
y = data[:, -1]   # Cột cuối là nhãn (ung thư phổi)

# Chuẩn hóa dữ liệu X
X = normalize(X)

# Huấn luyện mô hình Logistic Regression
learning_rate = 0.01
num_iterations = 10000
beta, bias = logistic_regression(X, y, learning_rate, num_iterations)

# Dự đoán
predictions = predict(X, beta, bias)
print("Predictions:", predictions)
