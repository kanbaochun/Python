#coding:utf-8
#前向传播(两层神经网络)
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#定义输入和输出参数
x = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([1,1], stddev=1, seed=1))

#定义前向传播过程
y = tf.matmul(x, w1)

#用会话图输出计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print(sess.run(y, feed_dict={x:[[0.7]]}))
