
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


x,y=np.mgrid[-2:2:20j,-2:2:20j]
z=x*np.exp(-x**2-y**2) 
ax=plt.subplot(111)
ax.plot(x,y,z,alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')


plt.show()
