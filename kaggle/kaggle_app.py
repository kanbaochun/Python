from keras.models import Model, load_model
import shutil
import numpy as np
from keras.preprocessing import image

def pre_pic(fname):
	img = image.load_img(fname, target_size=(224, 224))
	img_array = image.img_to_array(img)
	img_ready = np.array(img_array)/255.
	img_ready = np.reshape(img_ready, (-1, 224, 224, 3))
	return img_ready

model = load_model('saved_models/kaggle_20v1_model.087.h5')
for i in [7651]:
	fname = 'test2/test/' + str(i) + '.jpg'
	img_ready = pre_pic(fname)
	a = model.predict(img_ready)
	print(a[0][0])
	# if a[0][0] > 0.5:
	# 	img_path = 'test2/dog/' + str(i) + '.jpg'
	# 	shutil.copy(fname, img_path)
	# else:
	# 	img_path = 'test2/cat/' + str(i) + '.jpg'
	# 	shutil.copy(fname, img_path)
	# if i % 100 == 0:
	# 	print('已复制{:.1f}%'.format(i/125))
