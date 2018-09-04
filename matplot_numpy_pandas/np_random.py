# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
#生成一个size为24的二维数组
c = np.arange(24).reshape(4,6)
#np.random.shuffle(c)乱序排列改变原数组
np.random.shuffle(c)
#d = np.random.permutation(c)乱序排列不改变原数组
d = np.random.permutation(c)
#np.random.randint(100,200,(8,))生成100-200的int类型的随机数组，（）内为数组形状
a = np.random.randint(100,200,(8,))
#np.random.choice(c,(3,2),replace=False)随机抽取元素，replace改变重复抽取规则
np.random.choice(a,(3,2),replace=False)
#np.random.uniform(0,10,(3,4))生成0-10随机数，均匀分布
e = np.random.uniform(0,10,(3,4))
#np.random.normal(10,5,(3,4))生成正态分布数组，均值是10，标准差为5,3*4数组
f = np.random.normal(10,5,(3,4))
#np.random.poisson(0.5,(3,4))0.5为随机事件发生概率
h = np.random.poisson(0.5,(3,4))
#np.random.rand(8,)生成括号形状的0-1随机数
g = np.random.rand(8,)
#np.random.randn(8,)生成括号形状的0-1随机数,正态分布
i = np.random.randn(8,)
#np.random.seed(10)设定随机数种子，设定种子后产生随机数与不变
np.random.seed(10)