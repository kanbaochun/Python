# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
from pylab import mpl
import pandas as pd

mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
d = pd.read_csv('01.csv', encoding='ANSI')
dl = d['公司地址'].value_counts()
fig = dl[:10].plot(kind='bar').get_figure()
plt.show()





