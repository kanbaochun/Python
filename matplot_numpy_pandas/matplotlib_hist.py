# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
#直方图y轴为元素出现概率，x轴为元素值
np.random.seed(0)
mu, sigma = 100, 20 #设置均值和方差
a = np.random.normal(mu, sigma, size=100)
plt.hist(a, 40, normed=1, histtype='stepfilled', facecolor='brown', 
         alpha=0.75)
plt.title('Histogram')

plt.show()
 