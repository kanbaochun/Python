import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward_lenet5
import numpy as np
import os

#定义常量
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'


def backward(mnist):

	#训练集占位
	x = tf.placeholder(tf.float32, 
		[BATCH_SIZE, 
		mnist_forward_lenet5.IMAGE_SIZE,
		mnist_forward_lenet5.IMAGE_SIZE,
		mnist_forward_lenet5.NUM_CHANNELS])
	y_ = tf.placeholder(tf.float32, shape=(None, mnist_forward_lenet5.OUTPUT_NODE))
	#前向传播
	y = mnist_forward_lenet5.forward(x, True, REGULARIZER)
	#轮数计数器
	global_step = tf.Variable(0, trainable=False)

	#损失函数
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, 
		labels=tf.argmax(y_, 1))
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses'))

	#定义指数学习率
	learning_rete = tf.train.exponential_decay(LEARNING_RATE_BASE,
		    	global_step,
		    	mnist.train._num_examples/BATCH_SIZE, 
		    	LEARNING_RATE_DECAY, 
		    	staircase=True)

	#定义训练方法
	train_step = tf.train.AdamOptimizer(learning_rete).minimize(loss, 
		global_step=global_step)

	#定义滑动平均
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name='train')

	#实例化saver
	saver = tf.train.Saver()

	#创建会话图进行
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		#断点续训
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		#开始训练	
		for i in range(STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			reshaped_xs = np.reshape(xs, 
					(BATCH_SIZE, 
					mnist_forward_lenet5.IMAGE_SIZE,
					mnist_forward_lenet5.IMAGE_SIZE,
					mnist_forward_lenet5.NUM_CHANNELS))
			train_op_v, loss_v, global_step_v = sess.run([train_op, loss, global_step], 
				feed_dict={x:reshaped_xs, y_:ys})
			if i % 100 == 0:
				print('After %d steps, loss_total is %f' % (global_step_v, loss_v))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 
					global_step=global_step)
	
def main():
	mnist = input_data.read_data_sets('data/', one_hot=True)
	backward(mnist)

#调用函数运行
if __name__ == '__main__':
	main()


