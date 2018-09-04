import os
import h5py
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import image

#图片转化为数组
def pre_pic(fname):
	img = image.load_img(fname, target_size=(224, 224))
	img_array = image.img_to_array(img)
	img_ready = np.array(img_array)/255.
	return img_ready

#将数据写入h5
def pre_data():	
	h5f = h5py.File('x_train.h5', 'w')
	x_dataset = h5f.create_dataset('x_train', 
								(25000, 224, 224, 3),
								dtype='float32')
	y_dataset = h5f.create_dataset('y_train', 
								(25000, 1),
								dtype='float32')
	count = 0
	paths = ['train/cat', 'train/dog']
	for path in paths:
		files = os.listdir(path)
		for file in files:
			fname = path + '/' + file
			img_ready = pre_pic(fname)
			x_dataset[count] = img_ready
			if path == 'train/cat':
				y_dataset[count] = np.array(0)
			else:
				y_dataset[count] = np.array(1)
			if count % 100 == 0:
				print(str(count) + ' pictures having been transformed')
			count = count + 1
	h5f.close()

#定义主函数
if __name__ == '__main__':
	pre_data()

