import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BATCH_SIZE = 8
STEPS = 5000

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
datasets_X = np.array(datasets_X).reshape([-1,1])
datasets_Y = np.array(datasets_Y)

x = tf.placeholder(tf.float32, shape=(None,1))
y_ = tf.placeholder(tf.float32, shape=(None,1))

#设置神经网络
w = tf.Variable(tf.random_normal([1,1], stddev=1, seed=1))
b = tf.Variable(tf.constant(0.1, shape=(1,1)))
y = tf.matmul(x, w) + b
	

#定义损失函数
loss = tf.reduce_mean(tf.square(y-y_))
#指定训练算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 44
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:datasets_X[start:end], 
			y_:datasets_Y[start:end]})
		if i % 500 == 0:
			loss_v = sess.run(loss, feed_dict = {x: datasets_X, y_:datasets_Y})
			print('After %d steps, loss is %f' % (i, loss_v))	
			print(sess.run(w))
			print(sess.run(b))
