import numpy as np
import matplotlib.pyplot as plt  # Thư viện để vẽ đồ thị

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._sign
        self.weights = None
        self.bias = None

    def _sign(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 5

        print(np.zeros(n_features))

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                if y[idx] * y_predicted <= 0:
                    self.weights += self.lr * y[idx] * x_i
                    print('weights:',self.weights)
                    self.bias += self.lr * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)

    def plot_decision_boundary(self, X, y):
        # Thiết lập đồ thị
        fig, ax = plt.subplots()

        # Vẽ các điểm dữ liệu
        for idx, point in enumerate(X):
            if y[idx] == 1:
                ax.scatter(point[0], point[1], marker='o', color='blue', label='Class 1' if idx == 0 else "")
            else:
                ax.scatter(point[0], point[1], marker='x', color='red', label='Class -1' if idx == 0 else "")

        # Tính toán đường phân chia (decision boundary)
        x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        x2 = -(self.weights[0] * x1 + self.bias) / self.weights[1]  # y = -(w1*x1 + b) / w2

        # Vẽ đường phân chia
        ax.plot(x1, x2, color='green', label='Decision Boundary')

        # Thiết lập tiêu đề và nhãn
        ax.set_title('Perceptron Decision Boundary')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()

        # Hiển thị đồ thị
        plt.show()

# Hàm tạo dữ liệu ngẫu nhiên
def create_data(n_points=100):
    np.random.seed(42)  # Để đảm bảo tính tái lập của kết quả
    X = np.random.randn(n_points, 2)  # Tạo 100 điểm 2D ngẫu nhiên
    # Nhãn được sinh ra dựa trên một đường thẳng y = x
    y = np.where(X[:, 1] > X[:, 0], 1, -1)  # Nếu y > x thì nhãn là 1, ngược lại là -1
    return X, y

# Vẽ các điểm dữ liệu
def plot_data(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='Class 1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='Class -1')
    plt.title('Generated Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Tạo và hiển thị dữ liệu
if __name__ == "__main__":
    X, y = create_data(100)
    # plot_data(X, y)

    # Bạn có thể in ra dữ liệu nếu cần
    print("Dữ liệu đầu vào X:")
    print(X)
    print("Nhãn tương ứng y:")
    print(y)


# Example usage:
if __name__ == "__main__":
    X, y = create_data(100)
    X = np.array(X)
    y = np.array(y)  # Labels

    perceptron = Perceptron(learning_rate=0.1, n_iters=10)
    perceptron.fit(X, y)

    predictions = perceptron.predict(X)
    print("Predictions:", predictions)
    print("Weights:", perceptron.weights)
    print("Bias:", perceptron.bias)

    perceptron.plot_decision_boundary(X, y)
