from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
import json

(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
X_train = X_train_image.reshape(60000, 784).astype('float32')
X_test = X_test_image.reshape(10000, 784).astype('float32')
X_train_normalize = X_train/255
X_test_normalize = X_test/255
y_train_onehot = np_utils.to_categorical(y_train_label)
y_test_onehot =  np_utils.to_categorical(y_test_label)

#占位
inputs = Input(shape=(784,))
#输入值
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, output=predictions)
model.compile(optimizer='rmsprop', 
				loss='categorical_crossentropy',
				metrics=['accuracy'])

model.summary()
#模型加载
try:
	model.load_weights('model/mnist.h5')
	print('模型加载成功！继续训练模型')
except:
	print('模型加载失败，重新训练模型')

train_history = model.fit(X_train_normalize, y_train_onehot, validation_split=0.2, 
							epochs=5, batch_size=200, verbose=1)

model.save_weights('model/cifar10.h5')
print('训练模型已保存')

c = model.predict_on_batch(X_test_normalize[0].reshape(-1, 784))
print(np.argmax(c))
print(np.argmax(y_test_onehot[0]))

#保存训练信息
with open('model/mnist_model.json', 'w') as f:
	json.dump(train_history.history, f)
	print('保存成功')