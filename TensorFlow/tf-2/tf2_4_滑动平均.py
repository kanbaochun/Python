#coding:utf-8
#反向传播(两层神经网络)
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

w1 = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)
MOVEING_AVERAGE_DECAY = 0.99 #滑动平均衰减率


ema = tf.train.ExponentialMovingAverage(MOVEING_AVERAGE_DECAY, global_step)
ema_op = ema.apply(tf.trainable_variables()) #所有待优化参数求滑动平均


#用会话图输出计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)		
	sess.run(tf.assign(global_step, 100))
	sess.run(tf.assign(w1,10))
	for i in range(100):
		sess.run(ema_op)
		print(sess.run([w1, ema.average(w1)]))