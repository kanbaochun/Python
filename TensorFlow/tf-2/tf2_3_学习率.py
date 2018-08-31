#coding:utf-8
#反向传播(两层神经网络)
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1

w = tf.Variable(tf.constant(5, dtype=tf.float32))

global_step = tf.Variable(0, trainable=False)
learn_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 
	LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
#定义损失函数及反向传播算法
loss = tf.square(w+1)
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, 
	global_step=global_step)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#用会话图输出计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 40
	for i in range(STEPS):
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print("After %d steps, w is: %f, loss is %f" % (i, w_val, loss_val))
			
