##Xây dựng hàm mất mát
#Tên hàm  calculateLoss(X, y, w)
#Với X, y lần lượt là dữ liệu X và y; w là hệ số
#Hàm trả về giá trị của hàm mất mát J tương ứng với tham số đầu vào ở trên

def CalculateLoss(X, y, w):
  m = y.shape[0]
  h = np.dot(X,w)
  J = (1/(2*m))*np.sum(np.square(h-y))
  return J