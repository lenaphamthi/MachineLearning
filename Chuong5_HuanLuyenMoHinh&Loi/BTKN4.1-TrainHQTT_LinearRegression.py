import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
#Bước 1: Đọc dữ liệu
D = np.loadtxt(os.path.join('ex1data2.txt'), delimiter=',')
#Bước 2: Điều chỉnh dữ liệu - do đây là mô hình HQTT nên chấp nhận scale cả vector y
scaler = MinMaxScaler()
scaler.fit(D)
D = scaler.transform(D)
#Bước 3: Phân chia train - test theo tỉ lệ 70% - 30%
X, y = D[:, :-1], D[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
random_state=15)
#Bước 4: Xác định k-fold
kf = KFold(n_splits=10)
#Bước 5: Huấn luyện mô hình
print('Huấn luyện mô hình LinearRegression với k-fold')
model = LinearRegression()
step = 0
for train2_index, val_index in kf.split(X = X_train,y = y_train):
    step = step + 1
    print('\tBước lặp huấn luyện thứ: ', step)
    w_opt = np.zeros((X.shape[1], 1))
    X_train2, X_val = X_train[train2_index], X_train[val_index]
    y_train2, y_val = y_train[train2_index], y_train[val_index]
    model.fit(X_train2, y_train2)
    print('\t\tĐánh giá mô hình trên tập dữ liệu validation')
    y_hat = model.predict(X_val)
    print('\t\t\tMSE: ', mean_squared_error(y_val, y_hat))
#Bước 6: Kiểm định mô hình với tập dữ liệu test
print('ĐÁNH GIÁ HIỆU NĂNG CỦA MÔ HÌNH TRÊN TẬP DỮ LIỆU TEST')
y_hat = model.predict(X_test)
print('\tMSE: ', mean_squared_error(y_test, y_hat))