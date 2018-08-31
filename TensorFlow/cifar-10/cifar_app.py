import tensorflow as tf
import cifar_backward
import cifar_forward
import numpy as np
from PIL import Image

#图片测试
def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, 
			[1, 
			cifar_forward.IMAGE_SIZE,
			cifar_forward.IMAGE_SIZE,
			cifar_forward.NUM_CHANNELS])
		y = cifar_forward.forward(x, True, None)
		preValue = tf.argmax(y, 1)

		variable_averages = tf.train.ExponentialMovingAverage(cifar_backward.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(cifar_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
				coord.request_stop()
				coord.join(threads)
			else:
				print('No checkpoint file found')
				return -1

#图片特征提取
def pre_pic(picName):
	tfrecord_path = 'cifar/test_jpg'
	writer = tf.python_io.TFRecordWriter(tfrecord_path)
	img = Image.open(picName)
	reIm = img.resize((32,32), Image.ANTIALIAS)
	img_raw = reIm.tobytes()
	example = tf.train.Example(features=tf.train.Features(feature={
						'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
						}))
	writer.write(example.SerializeToString())
	writer.close()
	fileName_queue = tf.train.string_input_producer([tfrecord_path])
	reader = tf.TFRecordReader()
	_,Serialized_example = reader.read(fileName_queue)
	features = tf.parse_single_example(Serialized_example, 
			features={
			'img_raw':tf.FixedLenFeature([], tf.string)
			})
	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img.set_shape([3072])
	img_ready = tf.cast(img, tf.float32)* (1./255)
	img_batch = tf.train.shuffle_batch([img_ready],
									batch_size = 1,
									num_threads = 2,
									capacity = 10,
									min_after_dequeue = 1)
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		img_arr = sess.run([img_batch])
		testPicArr = np.reshape(np.array(img_arr), [1,32,32,3])
		coord.request_stop()
		coord.join(threads)
	return testPicArr

def main():
	while True:
		print('please choose a file')
		picName = input()
		lst = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		try:
			img_ready = pre_pic(picName)
			preValue = restore_model(img_ready)
			print('the picture is {}'.format(lst[int(preValue)]))
		except:
			print('please choose correct file')
		if picName == 'quit':
			break

if __name__ == '__main__':
	main()




	