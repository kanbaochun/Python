#coding:utf-8
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 30
seed = 2

#生成数据集
rng = np.random.RandomState(seed)
X = rng.randn(300,2)
Y = [[int(x0*x0 + x1*x1 < 2)] for (x0, x1) in X]
X = np.vstack(X).reshape(-1,2)
Y = np.vstack(Y).reshape(-1,1)
Y_c = [['red' if y else 'blue'] for y in Y]

#定义神经网络
def get_weight(shape, regularizer):
	w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.constant(0.01, shape=shape))
	return b

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2,11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1)+1)

w2 = get_weight([11,1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2

#定义损失函数
loss_mse = tf.reduce_mean(tf.square(y - y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

'''
#定义反向传播方法
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)


with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	steps = 40000
	for i in range(steps):
		start = (i*BATCH_SIZE) % 300
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i % 2000 == 0:
			loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y})
			print('After %d steps, loss_mse is %f' % (i, loss_mse_v))
	xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	probs = sess.run(y, feed_dict={x:grid})
	probs = probs.reshape(xx.shape)

plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
'''
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_total)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	steps = 40000
	for i in range(steps):
		start = (i*BATCH_SIZE) % 300
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i % 2000 == 0:
			loss_mse_v = sess.run(loss_total, feed_dict={x:X, y_:Y})
			print('After %d steps, loss_mse is %f' % (i, loss_mse_v))
	xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	probs = sess.run(y, feed_dict={x:grid})
	probs = probs.reshape(xx.shape)
	print(probs)
	
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()