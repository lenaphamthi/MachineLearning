import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def readData(filename,folder='', delimiter= ","):
   D = np.loadtxt(os.path.join(folder, filename), delimiter=delimiter)
   X = D[:, :-1]
   y = D[:, -1]
   return X, y

def featureScaling(X_train, X_test):
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   return X_train, X_test

def crossValPredict(model, X_train, y_train, cv=10):
   scores = cross_val_predict(model, X_train, y_train, cv=10)
   print('Kết quả huấn luyên 10-fold cv')
   print('\t', scores)
   return scores

def main():
   #Bước 1: Đọc dữ liệu
    X, y = readData('ex2data2.txt')
    print(X.shape[1])
   #Bước 2: Phân chia train - test theo tỉ lệ 70% - 30%
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

   # Bước 3: Chuẩn hóa dữ liệu
   X_train, X_test = featureScaling(X_train, X_test)
   #Bước 4: Khởi tạo mô hình hồi quy logistic, với thuật toán tối ưu là liblinear
   #Bước lặp 1500; multi_class = 'auto' để tự phát hiện nhãn lớp nhị phân hay đa nhãn lớp
   classifier = LogisticRegression(solver='liblinear', max_iter=1500, multi_class='auto')

   #Bước 5: Huấn luyện mô hình cv = 10 và độ đo là scoring='accuracy' và lấy kết quả dự đoán về
   scores = crossValPredict(classifier, X_train, y_train, cv=10)

if __name__ == "__main__":
   main()
