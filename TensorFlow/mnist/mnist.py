import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data', one_hot=True)
import scipy.misc

save_dir = 'data/raw/'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for i in range(20):
	img_arr = mnist.train.images[i, :]
	img_mat = img_arr.reshape(28, 28)
	fname = save_dir + 'mnist_train_%d.jpg' % i
	print(fname)
	scipy.misc.toimage(img_mat, cmin=0.0, cmax=1.0).save(fname)




