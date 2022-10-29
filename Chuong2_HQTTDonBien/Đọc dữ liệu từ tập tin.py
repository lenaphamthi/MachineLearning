#Trích X, y ra từ dữ liệu gốc
#Chỉnh sửa X thêm 1 cột vector 1 vào bên trái
#Trả về X và y
#Hàm đặt tên là readData(folder, filename)
#folder và filename là 2 tham số kiểu string chứa đường dẫn thư mục và tên tập tin dữ liệu

import numpy as np
import os

def readData(filename,folder=""):
  data = np.loadtxt(os.path.join(folder,filename),delimiter=',')
  X= data[:,0]
  y = data[:,1]
  m = y.shape[0]
  X = np.stack([np.ones(m),X],axis=1)
  return X,y