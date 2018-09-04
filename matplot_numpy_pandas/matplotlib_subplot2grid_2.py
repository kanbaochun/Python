# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#(3,3)表示绘图区划分为3行3列
gs = gridspec.GridSpec(3,3)
a = np.arange(10).reshape(10,)
ax1 = plt.subplot(gs[0,:])
plt.plot(a)
ax1 = plt.subplot(gs[1,:-1])
plt.plot(a)
ax1 = plt.subplot(gs[1:,2])
plt.plot(a)
ax1 = plt.subplot(gs[2,0])
plt.plot(a)
ax1 = plt.subplot(gs[2,1])
plt.plot(a)
plt.show()