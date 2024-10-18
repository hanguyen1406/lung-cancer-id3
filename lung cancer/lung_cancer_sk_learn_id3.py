import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Tải dữ liệu
data = pd.read_csv('./data.csv')

# Khám phá dữ liệu
print(data.info())
print(data.describe())

# Tiền xử lý
X = data.drop('ung thư phổi', axis=1)  # Thay 'ung thư phổi' bằng tên cột nhãn thực tế
y = data['ung thư phổi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình ID3 (Cây quyết định)
model_id3 = DecisionTreeClassifier(criterion='entropy')  # Sử dụng ID3
model_id3.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model_id3.predict(X_test)
print(classification_report(y_test, y_pred))

