import numpy as np
import os

def readData(filename,folder=""):
    data = np.loadtxt(os.path.join(folder,filename),delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    m = X.shape[0]
    n = X.shape[1]
    X= np.reshape(X,(m,n))
    y = np.reshape(y,(m,1))
    #them 1 cot x0 = 1 vao x
    x0 = np.ones((m,1))
    X = np.column_stack([x0,X])
    return X, x0

def featureVectorScaling(data):
    avg =np.mean(data)
    sln = data.max()
    snn = data.min()
    data_scl = (data-avg)/(sln-snn)
    print(data_scl[1])
    return data_scl

def normalizeData(X):
    X_scl = X[:,0]
    for i in range(1,X.shape[1]):
        scl = featureVectorScaling(X[:,i])
        X_scl = np.column_stack([X_scl,scl])
    return X_scl

#Ham hw(x)
def sigmoid(X,w):
    result = 1/(1+np.exp(np.dot(X,w)))
    return result

def loss(X,y,w):
    m = y.shape[0]
    result = (-1/m)*np.sum(np.dot(y.T,np.log(sigmoid(X,w)))) + np.dot((1-y).T,np.log(1-sigmoid(X,w)))
    return result

def gradient(X,y,w):
    m = X.shape[0]
    result = (1/m)*np.dot(X.T,sigmoid(X,w)-y)
    return result

def gradientDescent(X,y,w,alpha,n_iters):
    w_optial = w.copy()
    J_histoty = []
    for i in range(n_iters):
        w_optial = w_optial - alpha*gradient(X,y,w_optial)
        J_histoty.append(loss(X,y,w_optial))
    return w_optial,J_histoty

def main():
    X,y = readData("ex2data1.txt")
    X_scl = normalizeData(X)
    n = X_scl.shape[1]
    w = np.zeros((n,1))
    alpha = 0.01
    n_iters = 2000
    w_opt, J_hist = gradientDescent(X_scl,y,w,alpha,n_iters)
    print("Ket qua la: ")
    print("Tong so w toi uu la: ",w_opt)
    print("Gia tri loss toi uu: ", J_hist[-1])

if __name__ == "__main__":
    main()