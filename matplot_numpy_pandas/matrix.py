from numpy import *

#创建矩阵
a = mat([[3,4],[2,16]])
#a的逆矩阵
b = linalg.inv(a)
#矩阵转置 
c = transpose(a) 
print(c)