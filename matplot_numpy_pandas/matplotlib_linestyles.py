# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

a = np.arange(10)
#x轴y轴，线的颜色和线型
plt.plot(a, 1.5*a, 'go--', a, 2.5*a, 'd', a, 3.5*a, 'rx', a, 4.5*a, 'b--')
plt.show()