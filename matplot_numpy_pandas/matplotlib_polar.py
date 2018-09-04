# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np

N = 20
theta = np.linspace(0.0, 2*np.pi, N, endpoint=False)
radii = 10*np.random.rand(N)
width = np.pi/4*np.random.rand(N)

ax = plt.subplot(111, projection='polar')               #绘图区设置为极坐标对象
bars = ax.bar(theta, radii, width=width, bottom=0.0)    #使用极坐标.bar方法

for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r/10.))
    bar.set_alpha(0.5)

plt.show()
