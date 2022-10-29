##Hàm gradientDescent(X, y, w, alpha, n), với:
#X, y là dữ liệu huấn luyện
#w là vector trọng số
#alpha là learning rate - 1 số thực
#n là số bước lặp
##Hàm trả về:
#Giá trị vector trọng số tối ưu tìm được theo thuật toán Gradient Descent - w_optimal
#List chứa tất cả các giá trị của hàm mất mát tương ứng với các giá trị vector trọng số tại mỗi bước lặp

def gradientDescent(X, y, w = np.zeros(2), alpha=0.01, n=1500):
  J_history = []
  w_optimal= w.copy()
  m = y.shape[0]
  for i in range(n):
    w_optimal = w_optimal- (alpha/m)*(np.dot(X,w_optimal)-y).dot(X)
  J_history.append(CalculateLoss(X, y, w_optimal))
  return w_optimal,J_history