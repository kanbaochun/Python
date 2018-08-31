from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import matplotlib.pyplot as plt 

def plot_image(image, label, prediction, idx, num=10):
	fig = plt.gcf()
	fig.set_size_inches(10, 6)
	if num > 25: num = 25
	for i in range(num):
		ax = plt.subplot(2 ,5, i + 1)
		ax.imshow(image[idx], cmap='binary')
		title = str(i) + '.' + label_dict[label[i][0]]
		if len(prediction) > 0:
			title += '=>' + ',prediction=' + label_dict[prediction[i]]
		ax.set_title(title, fontsize=10)  
		ax.set_xticks([]);ax.set_yticks([])
		idx += 1
	plt.show()

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

if __name__ == '__main__':
	(X_train_image, y_train_label), (X_test_image, y_test_label) = cifar10.load_data()
	X_train_image = X_train_image[:10]/255
	y_train_label = y_train_label[:10]
	
	label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 
				6:'frog', 7:'horse', 8:'ship', 9:'truck'}

	for x_batch,y_batch in datagen.flow(X_train_image,y_train_label,batch_size = 10):
		for i in range(10):
			pics_raw = x_batch[i]
			pics = array_to_img(pics_raw)
			ax = plt.subplot(10//5, 5, i+1)
			ax.axis('off')
			ax.set_title(label_dict[y_batch[i][0]])
			plt.imshow(pics)
		plt.show()
	
	
