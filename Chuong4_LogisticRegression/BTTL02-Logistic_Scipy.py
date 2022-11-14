import numpy as np
import os
from scipy import optimize

def readData(filename, folder=""):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:,:-1]
    y = data[:, -1]
    m = X.shape[0]
    n = X.shape[1]
    X = np.reshape(X, (m,n))
#Do w là vector hàng (n, ) nên khi nhân với X.w sẽ cho ra vector hàng (m, )
#Nên y giữ nguyên là vector hàng chứ không reshape thành vector cột dạng ma trận (m,1)
#y = np.reshape(y, (m,1))
#Them cot x0 = 1 vao X
    x0 = np.ones((m,1))
    X = np.column_stack([x0, X])
    return X, y

def featureVectorScaling(data):
    avg = np.mean(data)
    sln = data.max()
    snn = data.min()
    data_scl = (data - avg)/(sln - snn)
    print(data_scl[1])
    return data_scl

def normalizeData(X):
    X_scl = X[:, 0]
    for i in range(1, X.shape[1]):
        scl = featureVectorScaling(X[:, i])
        X_scl = np.column_stack([X_scl, scl])
    return X_scl

#Day chinh la ham  hw(X)
def sigmoid(X, w):
    result = 1/(1 + np.exp(-np.dot(X, w)))
    return result

#Chinh sua lai ham mat mat J(w) cho phu hop voi yeu cau cua ham minimize cua scipy
#Thứ tự xuất hiện của các tham số được đổi lại là w, X, y
def loss(w, X, y):
    m = X.shape[0]
#Sử dụng biến tạm h để giảm số lần gọi hàm sigmoid
    h = sigmoid(X, w)
    result = (-1 / m) * np.sum(np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    return result

def toi_uu_bang_scipy(X,y,w,n_iters):
#Thứ tự xuất hiện các tham số của hàm loss phải đổi lại theo thứ tự trong optimize.minimize là
#loss(w, X, y)
    result = optimize.minimize(fun=loss, x0=w, args=(X,y),
                               method='L-BFGS-B',
                               options={"maxiter":n_iters} )
    w_optimal = result.x
    J_optimal = result.fun
    return w_optimal, J_optimal

def main():
    X, y = readData('ex2data1.txt')
    X = normalizeData(X)
    n = X.shape[1]
#Lưu ý: w trong thuật toán của scipy là vector hàng tương ứng 1d-array trong numpy
    w = np.zeros(n)
    n_iters = 2000
    w_opt, J_opt = toi_uu_bang_scipy(X,y,w, n_iters)
    print("Ket qua la: ")
    print('\t\tTrong so w toi uu la: ', w_opt)
    print('\t\tGia tri Loss toi uu: ', J_opt)

if __name__ == "__main__":
    main()