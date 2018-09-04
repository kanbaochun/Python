# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
x = np.arange(100).reshape(5,20)
#np写入文件，fmt为输出文件格式，delimiter为字符分割形式
np.savetxt('01.txt',x,fmt='%.1f',delimiter=',')
#np读取文件
b = np.loadtxt('01.txt', delimiter=',')
#多维读取
c = np.fromfile('wf_template.txt', dtype=float, count=-1, sep='' )
#读取文件a，b为两列
a,b = np.genfromtxt('wf_template.txt').transpose()
