import tensorflow as tf
import mnist_backward_lenet5
import mnist_forward_lenet5
import numpy as np
from PIL import Image

#图片测试
def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, 
			[1, 
			mnist_forward_lenet5.IMAGE_SIZE,
			mnist_forward_lenet5.IMAGE_SIZE,
			mnist_forward_lenet5.NUM_CHANNELS])
		y = mnist_forward_lenet5.forward(x, True, None)
		preValue = tf.argmax(y, 1)

		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward_lenet5.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(mnist_backward_lenet5.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				testPicArr = np.reshape(testPicArr, [1,28,28,1])
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print('No checkpoint file found')
				return -1
#图片特征提取
def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28), Image.ANTIALIAS)
	im_arr = np.array(reIm.convert('L'))
	threshold = 50
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
			if im_arr[i][j] < threshold:
				im_arr[i][j] = 0
			else:
				im_arr[i][j] = 255

	nm_arr = im_arr.reshape([1,784])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0/255)

	return img_ready


def main():
	while True:
		print('please choose a file')
		picName = input()
		try:
			img_ready = pre_pic(picName)
			preValue = restore_model(img_ready)
			print('The number is %d' % (preValue))
		except:
			print('please choose correct file')
		if picName == 'quit':
			break

if __name__ == '__main__':
	main()