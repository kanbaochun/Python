import numpy as np

seed = 2
#生成数据集
def generateds():
	rng = np.random.RandomState(seed)
	X = rng.randn(300,2)
	Y_ = [[int(x0*x0 + x1*x1 < 2)] for (x0, x1) in X]
	X = np.vstack(X).reshape(-1,2)
	Y_ = np.vstack(Y_).reshape(-1,1)
	Y_c = [['red' if y else 'blue'] for y in Y_]
	return X, Y_, Y_c
