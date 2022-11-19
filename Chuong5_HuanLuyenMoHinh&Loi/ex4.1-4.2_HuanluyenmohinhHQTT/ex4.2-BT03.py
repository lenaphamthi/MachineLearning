import os
import numpy as np
from scipy import optimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def readData(filename,folder=''):
    data = np.loadtxt(os.path.join(folder, filename), delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis=1)
    return X, y


def normScaling(X, y):
    for col in range(1, X.shape[1]):
        temp = X[:, col]
        X[:, col] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
    return X, y


def standardScaling(X, y):
    for col in range(1, X.shape[1]):
        temp = X[:, col]
        X[:, col] = (temp - np.mean(temp)) / (np.std(temp))
    return X, y


def predict(X, w):
    h_w = 1 / (1 + np.exp(- np.dot(X, w)))
    return (h_w >= 0.5).astype('int32')


def costFunction(w, X, y):
    m = X.shape[0]
    h_w = 1 / (1 + np.exp(- np.dot(X, w)))
    J_w = (-1 / m) * (np.dot(y, np.log(h_w)) + np.dot((1 - y), np.log(1 - h_w)))
    return J_w


def logisticRegression(X, y, w_init, method, iterations):
    result = optimize.minimize(fun=costFunction, x0=w_init, args=(X, y),
                               method=method,
                               options={"maxiter": iterations})
    return result.x, result.fun


def compareAlgorithms(X_train, y_train, X_test, y_test, algorithms):
    w_init = np.zeros((X_train.shape[1], 1))
    result = {}
    for algorithm in algorithms:
        w, loss = logisticRegression(X_train, y_train, w_init, method=algorithm, iterations=1500)
        y_pred = predict(X_test, w)
        result[algorithm] = accuracy_score(y_pred, y_test)
    return result


def main():
    # Đọc dữ liệu
    X, y = readData('ex2data1.txt')

    # Tách dữ liệu thành training set và test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=5)

    # Chuẩn hóa dữ liệu
    X_train, y_train = standardScaling(X_train, y_train)
    X_test, y_test = standardScaling(X_test, y_test)

    # Huấn luyện mô hình và so sánh mô hình
    '''Other algorithms:
    TNC, BFGS, L-BFGS-B, Nelder-Mead, Powell, CG, Newton-CG, COBYLA, SLSQP, ...'''
    comparison = compareAlgorithms(X_train, y_train, X_test, y_test, ['TNC', 'BFGS', 'L-BFGS-B'])
    print(comparison)


if __name__ == "__main__":
    main()