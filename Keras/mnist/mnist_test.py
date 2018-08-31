import numpy as np 
import pandas as pd 
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
import matplotlib.pyplot as plt

def plot_image(image, label, prediction, idx, num):
	fig = plt.gcf()
	fig.set_size_inches(8, 8)
	if num > 25: num = 25
	for i in range(num):
		ax = plt.subplot(2 ,5, i + 1)
		ax.imshow(image[idx], cmap='binary')
		title = 'label=' + str(label[idx])
		ax.set_title('label = ' + str(label[i]))
		if len(prediction) > 0:
			title += ',prediction=' + str(prediction[idx])
		ax.set_title(title, fontsize=10)  
		ax.set_xticks([]);ax.set_yticks([])
		idx += 1
	plt.show()

if __name__ == '__main__':
	(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
	plot_image(X_train_image, y_train_label, [], 0, 10)






