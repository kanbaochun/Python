import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

img_ready = pre_pic('pic/5.png')
img_mat = img_ready.reshape(28, 28)
plt.imshow(img_mat)
plt.show()

'''
for i in range(10):
	fname = 'pic' + '/' + str(i) + '.png'
	img_ready = pre_pic(fname)
	img_lst = list(img_ready[0])
	txtName = 'txt' + '/' + str(i) + '.txt'
	with open(txtName, 'w') as f:
		for i in range(len(img_lst)):
			if i % 28 == 0:
				if i != 0:
					f.write('\t\n')
			else:
				f.write(str(int(img_lst[i])))
	f.close()
	'''