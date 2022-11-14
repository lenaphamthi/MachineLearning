import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
D = np.loadtxt(os.path.join("ex2data1.txt"),delimiter=",")
print('Kích thước của tập dữ liệu: ', D.shape)
print('Lấy ra tập dữ liệu X, y')
X, y = D[:,:-1], D[:, -1]
print('Thực hiện MinMaxScaler')
#Khởi tạo bộ điều chỉnh dữ liệu
scaler = MinMaxScaler()
#Phải thực hiện thao tác fit(data) trước khi điều chỉnh dữ liệu
scaler.fit(X)
#Thực hiện điều chỉnh dữ liệu trên X, không điều chỉnh với y
X = scaler.transform(X)
print('Kích thước tập X: ', X.shape)
print('Kích thước vector y: ', y.shape)
print('Huấn luyện mô hình LogisticRegression')
#Khởi tạo mô hình
model = LogisticRegression()
#Huấn luyện mô hình với tập dữ liệu X, y
model.fit(X,y)
#Bộ trọng số tối ưu:
print('\t\tw optimal: ', model.coef_)