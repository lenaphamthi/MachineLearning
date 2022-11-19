import numpy as np

def read_txt(filename, delimiter):
    data = np.loadtxt(filename, delimiter = delimiter)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def split_train_test(X, y, test_size, random_state = None):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test

def print_train_test(X_train, X_test, y_train, y_test):
    print('Kích thước dữ liệu train - test:')
    print("\t\tX train: ", X_train.shape, "; y train: ",y_train.shape)
    print("\t\tX test: ", X_test.shape, "; y test: ",y_test.shape)
def main():
    X, y = read_txt("ex1data2.txt", delimiter=',')
    X_train, X_test, y_train, y_test =  split_train_test(X, y, test_size = 0.3, random_state=4)
    print_train_test(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()