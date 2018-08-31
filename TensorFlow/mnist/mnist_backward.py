import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import mnist_forward


#定义常量
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'


def backward(mnist):

	#训练集占位
	x = tf.placeholder(tf.float32, shape=(None, mnist_forward.INPUT_NODE))
	y_ = tf.placeholder(tf.float32, shape=(None, mnist_forward.OUTPUT_NODE))
	#前向传播
	y = mnist_forward.forward(x, REGULARIZER)
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
	with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		#断点续训
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		#开始训练	
		for i in range(STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			train_op_v, loss_v, global_step_v = sess.run([train_op, loss, global_step], 
				feed_dict={x:xs, y_:ys})
			if i % 1000 == 0:
				print('After %d steps, loss_total is %f' % (global_step_v, loss_v))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 
					global_step=global_step)
	
def main():
	mnist = input_data.read_data_sets('data', one_hot=True)
	backward(mnist)

#调用函数运行
if __name__ == '__main__':
	main()


