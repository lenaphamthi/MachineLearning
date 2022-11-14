import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
D = np.loadtxt(os.path.join("ex1data2.txt"),delimiter=",")
print('Kích thước của tập dữ liệu: ', D.shape)
print('Giá trị của tập dữ liệu: ')
print(D)
print('Thực hiện MinMaxScaler')
#Khởi tạo bộ điều chỉnh dữ liệu
scaler = MinMaxScaler()
#Phải thực hiện thao tác fit(data) trước khi điều chỉnh dữ liệu
scaler.fit(D)
#Thực hiện điều chỉnh dữ liệu
D = scaler.transform(D)
print('Kích thước của tập dữ liệu: ', D.shape)
print('Giá trị của tập dữ liệu: ')
print(D)
print('Lấy ra tập dữ liệu X, y')
X, y = D[:,:-1], D[:, -1]
print('Kích thước tập X: ', X.shape)
print('Kích thước vector y: ', y.shape)