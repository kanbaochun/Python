from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers
import json
import matplotlib.pyplot as plt
import pandas as pd 

#数据集载入及预处理
(X_train_image, y_train_label), (X_test_image, y_test_label) = cifar10.load_data()
X_train_normalize = X_train_image.astype('float32')/255.
X_test_normalize = X_test_image.astype('float32')/255.
y_train_onehot = np_utils.to_categorical(y_train_label)
y_test_onehot =  np_utils.to_categorical(y_test_label)

#数据增强
datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(X_train_normalize)
train_datagen = datagen.flow(X_train_normalize, y_train_onehot, batch_size=32)
#优化器
sgd = optimizers.SGD(lr=0.001, decay=0.999)
#卷积神经网络模型建立
model = Sequential()
#卷积与池化层1
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', 
				input_shape=(32, 32, 3), activation='relu',
				kernel_initializer='glorot_uniform', 
				bias_initializer='zeros'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
#卷积与池化层2
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
				kernel_initializer='glorot_uniform', 
				bias_initializer='zeros'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
# #卷积与池化层3
# model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
# 				kernel_initializer='glorot_uniform', 
# 				bias_initializer='zeros'))
# model.add(Dropout(0.25))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#平坦层(单个训练数据转换成一维)
model.add(Flatten())
model.add(Dropout(0.25))
#建立全连接(2个隐藏层)
# model.add(Dense(2048, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(512, activation='relu', kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
print(model.summary())

#模型加载
try:
	model.load_weights('model/cifar10.h5')
	print('模型加载成功！继续训练模型')
except:
	print('模型加载失败，重新训练模型')

#训练模型
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
train_history = model.fit_generator(generator=train_datagen, 
								epochs=300, verbose=1, 
								validation_data=(X_test_normalize, y_test_onehot))
#模型的保存
model.save_weights('model/cifar10.h5')
print('训练模型已保存')

#评估模型的准确率
scores = model.evaluate(X_test_normalize, y_test_onehot)
print()
print('accuracy = ', scores[1])
prediction = model.predict_classes(X_test_normalize)


#显示混淆矩阵
#pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])

#保存训练信息
with open('model/model.json', 'w') as f:
	json.dump(train_history.history, f)
	print('保存成功')



	




