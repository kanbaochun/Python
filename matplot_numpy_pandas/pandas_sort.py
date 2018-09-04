# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

b = pd.DataFrame(np.arange(20).reshape(4,5), index=['b','a','c','d'])
c = pd.Series(np.arange(4))
#默认在0轴（列，columns）进行运算
d = b-c   
#在1轴(行，index)进行运算
e = b.sub(c, axis=0)
#索引（0轴）排序
f = b.sort_index()
#行排序，axis=1为1轴，axis=0为0轴排序
g = b.sort_index(axis=1, ascending=False)
#按列（0轴）排序
h = f.sort_values(2, ascending=False) #DataFrame排序 
#按行排序
i = f.sort_values('b', axis=1,ascending=False)





