#coding:utf-8
#反向传播(两层神经网络)
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt 

#定义常数
BATCH_SIZE = 8
seed = 23455
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1

#生成虚拟数据集
rng = np.random.RandomState(seed) 
X = rng.rand(32)
Y = [[x + np.random.rand()/10] for x in X]
X = np.vstack(X).reshape([-1,1])
Y = np.vstack(Y).reshape([-1,1])

#定义输入和输出参数
x = tf.placeholder(tf.float32, shape=(None, 1))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w = tf.Variable(tf.random_normal([1,1], stddev=1, seed=1))
b = tf.Variable(tf.constant(0.1, shape=(1,1)))

#定义前向传播过程
y = tf.matmul(x, w) + b

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 
	LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
#定义损失函数及反向传播算法
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#用会话图输出计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 10001
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i % 1000 == 0:
			loss_v = sess.run(loss, feed_dict={x:X, y_:Y})
			print("After %d steps, loss is %f" % (i, loss_v))
			print('w is ',sess.run(w))
			print('b is ',sess.run(b))
	w = sess.run(w)
	b = sess.run(b)

x = np.arange(min(X), max(X), 0.1).reshape([-1,1])
plt.scatter(X, Y, c='blue')
plt.plot(x, w*x + b, c='red')
plt.show()


