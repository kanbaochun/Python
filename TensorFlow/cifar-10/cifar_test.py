import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
import cifar_forward
import cifar_backward
import cifar_tfrecord


#定义常量
TEST_INTERVAL_SECS = 3
test_num_examples = 1000

def test():

	#数据输入占位
	x = tf.placeholder(tf.float32, 
		[test_num_examples, 
		cifar_forward.IMAGE_SIZE,
		cifar_forward.IMAGE_SIZE,
		cifar_forward.NUM_CHANNELS])
	y_ = tf.placeholder(tf.float32, shape=(None, cifar_forward.OUTPUT_NODE))
	y = cifar_forward.forward(x, False, None)

	#加载滑动平均
	ema = tf.train.ExponentialMovingAverage(cifar_backward.MOVING_AVERAGE_DECAY)
	ema_restore = ema.variables_to_restore()
	saver = tf.train.Saver(ema_restore)

	#定义正确率统计函数
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#加载测试集数据
	img_batch, label_batch = cifar_tfrecord.get_tfrecord(test_num_examples, isTrain=False)

	while True:
		with tf.Session() as sess:

			#加载训练模型
			ckpt = tf.train.get_checkpoint_state(cifar_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

				#开启线程协调器
				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(sess=sess, coord=coord)

				#引入数据集开始测试
				xs, ys = sess.run([img_batch, label_batch])
				reshaped_x = np.reshape(xs, 
					(test_num_examples, 
					cifar_forward.IMAGE_SIZE,
					cifar_forward.IMAGE_SIZE,
					cifar_forward.NUM_CHANNELS))
				accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_x, 
					y_: ys})
				print('After %s training steps, test accuracy = %g' % 
					(global_step, accuracy_score))

				coord.request_stop()
				coord.join(threads)
			else:
				print('No checkpoint file found')
				return
			time.sleep(TEST_INTERVAL_SECS)

def main():
	test()

if __name__ == '__main__':
	main()