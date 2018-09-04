# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

a = np.arange(15).reshape(3,5)
#np.sum(a,axis = 0)求纵轴方向数组和
b = np.sum(a,axis = 0)
#np.mean(a,axis = 0)求纵轴方向数组平均值
c = np.mean(a,axis = 0)
#np.average(a,axis = 0,weights=(10,5,1))求纵轴方向数组加权平均值
d = np.average(a,axis = 0,weights=(10,5,1))
#np.unravel_index(np.argmax(a), a.shape)返回最大值得索引位置
e = np.unravel_index(np.argmax(b), b.shape)