# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

a = pd.Series([1,2,3,4],['a','b','c','d'])
#索引和切片类似字典，a.index，a.values
b = pd.DataFrame(np.arange(10).reshape(2, 5))
dt = {'one':pd.Series([1,2,3], index = ['a', 'b', 'c']), 
      'two':pd.Series([9,8,7,6], index = ['a', 'b', 'c', 'd'])}
c = pd.DataFrame(dt)

dl = {'城市':['北京','上海','广州','深圳', '沈阳'],
      '环比':[101.5, 101.2, 101.3, 102.0, 100.1],
      '同比':[120.7, 127.3, 119.4, 140.9, 101.4],
      '定基':[121.4, 127.8, 120.0, 145.5, 101.6],
      }

#d = pd.DataFrame(dl, index = ['c1','c2','c3','c4','c5'])
#d = d.reindex(columns=['城市','环比','同比','定基'])

#d['城市']         #列索引
#d.ix['c1']        #行索引
#删除列
#f = d.drop(['定基','环比'], axis=1)
#删除行
#g = d.drop(['c1', 'c2'])
#nc = d.columns.delete(2)
#ni = d.index.insert(0, 'c0')
#nd = d.reindex(index=ni, columns=nc, method='ffill')










