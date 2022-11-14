import os
import numpy as np
from matplotlib import pyplot as plt

def readData(filename,folder=""):
    data = np.loadtxt(os.path.join(folder, filename), delimiter=',')
    print('Original data shape', data.shape)
    X = data[:,:-1]
    print('X shape: ', X.shape)
    y = data[:,-1]
    print('y shape: ', y.shape)
    m = X.shape[0]
    print('Number of training examples m = ', m)
    x0 = np.ones((m,1))
    X = np.hstack([x0, X])
    print('Modified X shape: ', X.shape)
    y = np.reshape(y, (m,1))
    print('Modified y shape: ', y.shape)
    return X, y

def featureVectorScaling(data):
    snn = data.min()
    sln = data.max()
    data_scl = (data - snn)/(sln - snn)
    print(data_scl[1])
    return data_scl

def scaleData(X, y):
    X_scl = X[:, 0]
    for i in range(1, X.shape[1]):
        scl = featureVectorScaling(X[:, i])
        X_scl = np.column_stack([X_scl, scl])
    y_scl = featureVectorScaling(y)
    print('X_scl shape: ', X_scl.shape)
    print(X_scl[1,:])
    print('y scl shape: ', y_scl.shape)
    print(y_scl[1,:])
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
    print('w_optimal shape: ', w_optimal.shape)
    for i in range(n):
        h = np.dot(X, w_optimal)
        error = h - y
        w_optimal = w_optimal - (alpha/m)*np.dot(X.T, error)
        J_history.append(computeLoss(X, y, w_optimal))
    return w_optimal, J_history

def visualizeDataAndModel(X, y, w_optimal):
    fig = plt.figure()
    plt.plot(X[:,1], y, 'g^')
    plt.plot(X[:, 1], np.dot(X, w_optimal), 'r-')
    plt.legend(['Raw Data', 'Linear regression'])
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.show()

def main():
    n = 1500
    alpha = 0.01
    X, y = readData("ex1data2.txt")
    X_scl, y_scl = scaleData(X, y)
    print('X scl: ', X_scl[1,:])
    print('y scl: ', y_scl[1])
    w = np.zeros((X_scl.shape[1], 1))
    w, J_history = gradientDescent(X_scl, y_scl, w, alpha, n)
    print("Trong luong toi uu la: ", w)
    print("Loss function: ", J_history[-1])

if __name__ == '__main__':
    main()