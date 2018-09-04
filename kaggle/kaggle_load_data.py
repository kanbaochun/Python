import h5py

def load_data():
	h5f = h5py.File('dataset/kaggle_train.h5', 'r')
	x_train = h5f['x_train']
	y_train = h5f['y_train']
	return x_train, y_train