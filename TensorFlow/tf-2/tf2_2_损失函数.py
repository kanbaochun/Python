#coding:utf-8
#反向传播(两层神经网络)
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#定义常数
BATCH_SIZE = 8
seed = 23455
COST = 1
PROFIT = 9

#生成虚拟数据集
rng = np.random.RandomState(seed) 
X = rng.rand(32, 2)
Y = [[x1 + x2 + (rng.rand()/10.0 - 0.05)] for (x1, x2) in X]

#定义输入和输出参数
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))

#定义前向传播过程
y = tf.matmul(x, w1)

#定义损失函数及反向传播算法
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#用会话图输出计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 3001
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i % 1000 == 0:
			print("After %d steps, w1 is: \n" %(i), sess.run(w1))
			
