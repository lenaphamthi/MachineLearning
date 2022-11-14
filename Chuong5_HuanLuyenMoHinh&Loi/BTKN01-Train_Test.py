import os
import numpy as np
from sklearn.model_selection import train_test_split
D = np.loadtxt(os.path.join("ex1data2.txt"),delimiter=",")
X, y = D[:,:-1], D[:,-1]
print('Kích thước dữ liệu gốc:')
print("\t\tX: ", X.shape, "; y: ", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=15)
print('Kích thước dữ liệu train - test:')
print("\t\tX train: ", X_train.shape, "; y train: ",y_train.shape)
print("\t\tX test: ", X_test.shape, "; y test: ",y_test.shape)
