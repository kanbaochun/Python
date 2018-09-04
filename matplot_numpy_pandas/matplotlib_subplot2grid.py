# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

#(3,3)表示3行3列，（0,0）表示选中0行0列，colspan=3把第0行0列扩展至3列
plt.subplot2grid((3,3),(0,0),colspan=3)
a = np.arange(10).reshape(10,)
plt.plot(a)
plt.subplot2grid((3,3),(1,0),colspan=2)
plt.plot(a)
plt.subplot2grid((3,3),(1,2),rowspan=2)
plt.plot(a)
plt.subplot2grid((3,3),(2,0),colspan=1)
plt.plot(a)
plt.subplot2grid((3,3),(2,1),colspan=1)
plt.plot(a)
plt.show()