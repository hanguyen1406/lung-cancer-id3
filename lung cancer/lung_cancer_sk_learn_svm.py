import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Bước 1: Tải dữ liệu
data = pd.read_csv('data.csv')

# Bước 2: Khám phá dữ liệu
print(data.info())
print(data.describe())

# Bước 3: Tiền xử lý
X = data.drop('ung thư phổi', axis=1)  # Thay 'target_column' bằng tên cột nhãn
y = data['ung thư phổi']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 4: Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bước 5: Huấn luyện mô hình SVM
model = SVC(kernel='poly')  # Bạn có thể thử nghiệm với các kernel khác như 'rbf', 'poly', etc.
model.fit(X_train, y_train)

# Bước 6: Dự đoán và đánh giá
y_pred = model.predict(X_test)

# Bước 7: In kết quả đánh giá
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
