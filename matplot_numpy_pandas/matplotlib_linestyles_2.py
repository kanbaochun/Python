# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#负号正常显示
matplotlib.rcParams['axes.unicode_minus']=False

a = np.arange(0.0, 5.0, 0.02)

plt.title(r'余弦波曲线$y=cos(2\pi x)$',fontproperties = 'SimHei', fontsize=16) #列表头部
plt.xlabel('时间', fontproperties = 'SimHei', fontsize=16)
plt.ylabel('振幅', fontproperties = 'SimHei', fontsize=16)
#plt.text(2, 1, r'$\mu=100$', fontsize=15)                #标记某个点的值
plt.grid(True)                                            #次要网格线
plt.axis([-1, 6, -2, 2])                                  #设置x和y轴范围
plt.annotate(r'$mu=100$', xy=(2, 1), xytext=(3, 1.5), 
             fontproperties = 'SimHei', fontsize=12,
             arrowprops=dict(facecolor='black', shrink=0.05,width=2) )
plt.plot(a, np.cos(2*a*np.pi), 'r--')
plt.show()