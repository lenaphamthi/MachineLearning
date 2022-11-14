import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def readData(filename,folder=""):
    data = np.loadtxt(os.path.join(folder, filename), delimiter = ',')
    X = data[:,:-1]
    y = data[:, -1]
    m = X.shape[0]
    n = X.shape[1]
    X = np.reshape(X, (m,n))
    y = np.reshape(y, (m,1))
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

def loss(X, y, w):
    m = y.shape[0]
    result = (-1/m)*np.sum(np.dot(y.T, np.log(sigmoid(X, w))) + np.dot((1 - y).T, np.log(1 - sigmoid(X, w))))
    return result

def gradient(X, y, w):
    m = X.shape[0]
    result = (1/m)*np.dot(X.T, sigmoid(X, w) - y)
    return result

def gradientDescent(X, y, w, alpha, n_iters):
    w_optimal = w.copy()
    J_history = []
    for i in range(n_iters):
        w_optimal = w_optimal - alpha*gradient(X, y, w_optimal)
        J_history.append(loss(X, y, w_optimal))
    return w_optimal, J_history

#Hàm dự đoán nếu y_pred >=0.5 làm tròn thành 1, ngược lại là 0
def predict(y_pred):
    return np.rint(y_pred)

def acc_score(y, y_hat):
    m = y.shape[0]
    result = (1/m)*np.sum(y == y_hat)
    return  result

def main():
    X, y = readData('ex2data1.txt')
    X = normalizeData(X)
    n = X.shape[1]
    w = np.zeros((n, 1))
    alpha = 0.01
    n_iters = 2000
    #Chia train - test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.30,
                                                        random_state=15)
    w_opt, J_hist = gradientDescent(X_train, y_train, w, alpha, n_iters)
    print("Ket qua huan luyen mo hinh la: ")
    print('\t\tTrong so w toi uu la: ', w_opt)
    print('\t\tGia tri Loss toi uu: ', J_hist[-1])
    print('Ket qua du doan cua mo hinh')
    y_hat = predict(sigmoid(X_test, w_opt))
    print('\t\tMột số kết quả dự đoán: ', y_hat[:5,:])
    print('\t\tChỉ số Accuracy: ', acc_score(y_test, y_hat))
    print('\t\tSử dụng sklearn, Acc: ', accuracy_score(y_test.flatten(), y_hat.flatten()))

if __name__ == "__main__":
    main()