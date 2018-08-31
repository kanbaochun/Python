from PIL import Image
import numpy as np

picName = 'cifar/1.jpg'
img = Image.open(picName)
reIm = img.resize((32,32), Image.ANTIALIAS)
im_arr = np.array(reIm.convert('L'))
nm_arr = im_arr.reshape([1,1024])

	