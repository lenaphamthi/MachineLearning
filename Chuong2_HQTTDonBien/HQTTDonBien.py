import numpy as np
import os

def readData(filename,folder=""):
  data = np.loadtxt(os.path.join(folder,filename),delimiter=',')
  X= data[:,0]
  y = data[:,1]
  m = y.shape[0]
  X = np.stack([np.ones(m),X],axis=1)
  return X,y

def CalculateLoss(X, y, w):
  m = y.shape[0]
  h = np.dot(X,w)
  J = (1/(2*m))*np.sum(np.square(h-y))
  return J

def gradientDescent(X, y, w = np.zeros(2), alpha=0.01, n=1500):
  J_history = []
  w_optimal= w.copy()
  m = y.shape[0]
  for i in range(n):
    w_optimal = w_optimal- (alpha/m)*(np.dot(X,w_optimal)-y).dot(X)
  J_history.append(CalculateLoss(X, y, w_optimal))
  return w_optimal,J_history
def main():
  X,y = readData('ex1data1 (1).txt')
  w,J_history = gradientDescent(X,y)
  print("Giá trị vector trọng số tối ưu tìm được theo thuật toán Gradient Descent - w_optimal",w)
  print("List chứa tất cả các giá trị của hàm mất mát tương ứng với các giá trị vector trọng số tại mỗi bước lặp",J_history[-1])
if __name__ == "__main__":
  main()