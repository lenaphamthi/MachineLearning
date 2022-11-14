import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
D = np.loadtxt(os.path.join("ex1data2.txt"),delimiter=",")
print('Kích thước của tập dữ liệu: ', D.shape)
print('Thực hiện MinMaxScaler')
#Khởi tạo bộ điều chỉnh dữ liệu
scaler = MinMaxScaler()
#Phải thực hiện thao tác fit(data) trước khi điều chỉnh dữ liệu
scaler.fit(D)
#Thực hiện điều chỉnh dữ liệu
D = scaler.transform(D)
print('Lấy ra tập dữ liệu X, y')
X, y = D[:,:-1], D[:, -1]
print('Kích thước tập X: ', X.shape)
print('Kích thước vector y: ', y.shape)
print('Huấn luyện mô hình LinearRegression')
#Khởi tạo mô hình
model = LinearRegression()
#Huấn luyện mô hình với tập dữ liệu X, y
model.fit(X,y)
#Bộ trọng số tối ưu:
print('\t\tw optimal: ', model.coef_)