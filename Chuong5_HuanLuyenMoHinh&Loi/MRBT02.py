from sklearn.metrics import hamming_loss, accuracy_score
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def readData(filename, folder='', delimiter= ","):
   D = np.loadtxt(os.path.join(folder, filename), delimiter=delimiter)
   X = D[:, :-1]
   y = D[:, -1]
   return X,y

def featureScaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def KiemTraDLPhuThuoc(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    result = dict(zip(unique, counts))
    return result

def crossValScore(model, X_train, y_train, cv=10, scoring = 'accuracy'):
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
    print('Kết quả huấn luyên 10-fold cv')
    print('\t', scores)
    return scores

def main():
    #Bước 1: Đọc dữ liệu
    X, y = readData('ex2data2.txt')

    #Bước 2: Phân chia train - test theo tỉ lệ 70% - 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=15)

    # Bước 3: Chuẩn hóa dữ liệu
    X_train, X_test = featureScaling(X_train, X_test)

    #Số lượng k-fold được xác định tùy thuộc vào số lượng y_train
    result = KiemTraDLPhuThuoc(y_train)
    print(result)

    #Bước 4: Khởi tạo mô hình hồi quy logistic, với thuật toán tối ưu là liblinear
    #Bước lặp 1500; multi_class = 'auto' để tự phát hiện nhãn lớp nhị phân hay đa nhãn lớp
    classifier = LogisticRegression(solver='liblinear', max_iter=1500, multi_class='auto')

    #Bước 5: Huấn luyện mô hình cv = 10 và độ đo là scoring='accuracy' và in kết quả
    '''some other scores:
    balanced_accuracy, top_k_accuracy, average_precision, precision, recall, f1, ...'''
    scores = crossValScore(classifier, X_train, y_train, cv=10, scoring = 'precision')

if __name__ == "__main__":
    main()