import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def read_scale_data(filename,folder=""):
    D = np.loadtxt(os.path.join(folder, filename), delimiter=',')
    X, y = D[:,:-1], D[:,-1]
    X, y = scaleData(X, y)
    x0 = np.ones((X.shape[0], 1))
    X = np.column_stack([x0, X])
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.30, random_state=15)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    return X_train, X_test, y_train, y_test

def featureVectorScaling(data):
    avg = np.mean(data)
    sln = data.max()
    snn = data.min()
    data_scl = (data - avg)/(sln - snn)
    return data_scl

def scaleData(X, y):
    X_scl = X[:, 0]
    for i in range(1, X.shape[1]):
        scl = featureVectorScaling(X[:, i])
        X_scl = np.column_stack([X_scl, scl])
    y_scl = featureVectorScaling(y)
    return X_scl, y_scl

def computeLoss(X, y, w):
    m = y.shape[0]
    J = 0
    h = np.dot(X, w)
    J = (1/(2*m))*np.sum(np.square(h - y))
    return J

def gradientDescent(X, y, w, alpha, n):
    m = y.shape[0]
    J_history = []
    w_optimal = w.copy()
    for i in range(n):
        h = np.dot(X, w_optimal)
        error = h - y
        w_optimal = w_optimal - (alpha/m)*np.dot(X.T, error)
        J_history.append(computeLoss(X, y, w_optimal))
    return w_optimal, J_history

def mse(y, y_hat):
    m = y.shape[0]
    result = (1/m)*np.sum(np.square(y - y_hat))
    return result

def main():
    n = 1500
    alpha = 0.01
    X_train, X_test, y_train, y_test = read_scale_data("ex1data2.txt")
    X_train, y_train = scaleData(X_train, y_train)
    X_test, y_test = scaleData(X_test, y_test)
    print('Huấn luyện mô hình trên tập dữ liệu train')
    w = np.zeros((X_train.shape[1], 1))
    w, J_history = gradientDescent(X_train, y_train, w, alpha, n)
    print("\t\tOptimal weights are: ", w)
    print("\t\tLoss function: ", J_history[-1])
    print('Đánh giá mô hình trên tập dữ liệu test')
    y_hat = np.dot(X_test, w)
    print("\t\tMSE: ", mse(y_test, y_hat))
    print('\t\tSử dụng sklearn MSE: ', mean_squared_error(y_test, y_hat))
if __name__ == '__main__':
    main()