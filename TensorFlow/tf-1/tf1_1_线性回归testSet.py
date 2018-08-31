#coding:utf-8
#前向传播(两层神经网络)
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import matplotlib.pyplot as plt 

BATCH_SIZE = 11
seed = 23455
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1

# 读取数据集
datasets_X = []#房屋面积
datasets_Y = []#房屋价格
fr = open('houses.txt','r')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))#注意加上类型转换
    datasets_Y.append(int(items[1]))

#将datasets_X转换为二维数组，以符合 linear.fit 函数的参数要求
X = np.array(datasets_X).reshape([-1,1])
Y = np.array(datasets_Y).reshape([-1,1])

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
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#用会话图输出计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 20001
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 44
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

