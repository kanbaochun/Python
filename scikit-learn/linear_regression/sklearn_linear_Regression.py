# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# 读取数据集
datasets_X = []#房屋面积
datasets_Y = []#房屋价格
fr = open('houses.txt','r')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))#注意加上类型转换
    datasets_Y.append(int(items[1]))


#将datasets_X转换为二维数组，以符合 linear.fit 函数的参数要求
datasets_X = np.array(datasets_X).reshape([-1,1])
datasets_Y = np.array(datasets_Y)

#以数据datasets_X的最大值和最小值为范围，建立等差数列，方便后续画图。
minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX,maxX).reshape([-1,1])

'''
#线性回归
linear = linear_model.LinearRegression()  #建立线性回归模型
linear.fit(datasets_X, datasets_Y)        #数据训练   

# 图像中显示

plt.scatter(datasets_X, datasets_Y, color = 'red',label='origin data')
plt.plot(X, linear.predict(X), color = 'blue',label='linear regression prediction')
plt.legend()#使label生效
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
'''

#多项式回归
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(datasets_X)
#使用线性模型学习X_poly和datasets_Y之间的映射关系（即参数）
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, datasets_Y)

# 图像中显示
plt.scatter(datasets_X, datasets_Y, color = 'red',label='origin data')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue',label='Polynomial regression prediction')
plt.legend()#使label生效
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()