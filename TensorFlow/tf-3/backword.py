import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import generateds
import forward
import matplotlib.pyplot as plt

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01

def backward():

	#训练集占位
	x = tf.placeholder(tf.float32, shape=(None, 2))
	y_ = tf.placeholder(tf.float32, shape=(None, 1))

	#导入数据集
	X, Y_, Y_c = generateds.generateds()
	y = forward.forward(x, REGULARIZER)

	#轮数计数器
	global_step = tf.Variable(0, trainable=False)

	#定义指数学习率
	learn_rete = tf.train.exponential_decay(LEARNING_RATE_BASE,
		    	global_step,
		    	300/BATCH_SIZE, 
		    	LEARNING_RATE_DECAY, 
		    	staircase=True)

	#定义损失函数
	loss_mse = tf.reduce_mean(tf.square(y - y_))
	loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

	#定义训练方法
	train_step = tf.train.AdamOptimizer(learn_rete).minimize(loss_total)

	#创建会话图进行
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		for i in range(STEPS):
			start = (i*BATCH_SIZE) % 300
			end = start + BATCH_SIZE
			sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
			if i % 2000 == 0:
				loss_v = sess.run(loss_total, feed_dict={x:X, y_:Y_})
				print('After %d steps, loss_total is %f' % (i, loss_v))

		#生成网格坐标点		
		xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
		grid = np.c_[xx.ravel(), yy.ravel()]
		probs = sess.run(y, feed_dict={x:grid})
		probs = probs.reshape(xx.shape)

	#可视化训练集
	plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
	plt.contour(xx, yy, probs, levels=[.5])
	plt.show()

#调用函数运行
if __name__ == '__main__':
	backward()


