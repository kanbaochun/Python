import tensorflow as tf
from PIL import Image
import numpy as np 
import mnist_forward
import mnist_backward

#图片预处理
def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28, 28), Image.ANTIALIAS)	
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
	img_ready = np.multiply(nm_arr, 1.0/255.0)			
	return img_ready

#测试图片
def test_pic(img_ready):

	#预测操作方法
	with tf.Graph().as_default() as ng:
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y = mnist_forward.forward(x, None)
		preValue = tf.argmax(y, 1)

		#加载滑动平均
		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		#加载训练数据
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)

				#执行性预测操作
				preValue = sess.run(preValue, feed_dict={x:img_ready})
				return preValue
			else:
				print('No checkpoint file found')
				return -1
#定义预测函数
def main():
	print('please choose a handwriting number file')
	while 1:
		picName = input()
		if picName != 'quit':
			try:
				img_ready = pre_pic(picName)	
				preValue = test_pic(img_ready)
				print(preValue)
			except:
				print('Please enter the correct file address')
		else:
			break
		
if __name__ == '__main__':
	main() 

