#coding:utf-8
#反向传播(两层神经网络)
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt

#定义常数
BATCH_SIZE = 8

def loadDataSet():
	X = []; Y = []
	fr = np.loadtxt('testSet.txt',  delimiter='\t')
	for i in fr:
		X.append([float(i[0]), float(i[1])])
		Y.append(int(i[2]))
	X = np.array(X).reshape([-1,2])
	Y = np.array(Y).reshape([-1,1])
	Y_c = [['red' if y else 'blue'] for y in Y]
	return X, Y, Y_c

X, Y, Y_c = loadDataSet()

#定义输入和输出参数
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

#定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数及反向传播算法
loss = tf.reduce_mean(tf.square(y - y_))
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#用会话图输出计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 20001
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 100
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i%1000 == 0:
			total_loss = sess.run(loss, feed_dict={x:X, y_:Y})
			print("After %d train_step, total_loss is %g" % (i, total_loss))

	xx, yy = np.mgrid[-3:3:.01, -3:15:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	probs = sess.run(y, feed_dict={x:grid})
	probs = probs.reshape(xx.shape)
	print(probs)



plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
