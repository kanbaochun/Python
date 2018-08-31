from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add, Dropout
from keras.layers import AveragePooling2D, Input, Flatten, ZeroPadding2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.utils import np_utils
import json
import os

#定义残差块(skip 3 layers)
def identity_block(X, kernel_size, filters):
	X_shortcut = X
	X = BatchNormalization(epsilon=1e-8)(X)
	X = Activation('relu')(X)
	f1, f2, f3 = filters
	#改变通道数量
	X = Conv2D(filters=f1, kernel_size=(1, 1), 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal',
				kernel_regularizer=regularizers.l2(1e-4))(X)
	X = BatchNormalization(epsilon=1e-8)(X)
	X = Activation('relu')(X)
	#卷积提取信息操作
	X = Conv2D(filters=f2, kernel_size=kernel_size, 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal',padding='same',
				kernel_regularizer=regularizers.l2(1e-4))(X)
	X = BatchNormalization(epsilon=1e-8)(X)
	X = Activation('relu')(X)
	#改变通道数量
	X = Conv2D(filters=f3, kernel_size=(1, 1), 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal',padding='same',
				kernel_regularizer=regularizers.l2(1e-4))(X)
	#建立shortcut(添加残差块到主路径)
	X = Add()([X, X_shortcut])
	return X

#定义卷积块(skip 3 layers)
def convolution_block(X, kernel_size, filters, s=2, conv_first=True):
	X_shortcut = X
	f1, f2, f3 = filters
	#判断是否先先进行卷积
	if conv_first:
		X = BatchNormalization(epsilon=1e-8)(X)
		X = Activation('relu')(X)
	#改变通道数量
	X = Conv2D(filters=f1, kernel_size=(1, 1), 
				strides=(s, s), activation='relu', 
				kernel_initializer='he_normal',
				kernel_regularizer=regularizers.l2(1e-4))(X)
	X = BatchNormalization(epsilon=1e-8)(X)
	X = Activation('relu')(X)
	#卷积操作提取信息
	X = Conv2D(filters=f2, kernel_size=kernel_size, 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal',padding='same',
				kernel_regularizer=regularizers.l2(1e-4))(X)
	X = BatchNormalization(epsilon=1e-8)(X)
	X = Activation('relu')(X)
	##改变通道数量
	X = Conv2D(filters=f3, kernel_size=(1, 1), 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal',padding='same',
				kernel_regularizer=regularizers.l2(1e-4))(X)
	#shortcut path上的卷积操作
	X_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), 
				strides=(s, s), activation='relu', 
				kernel_initializer='he_normal',padding='same',
				kernel_regularizer=regularizers.l2(1e-4))(X_shortcut)
	#建立shortcut(添加残差块到主路径)
	X = Add()([X, X_shortcut])
	return X

#定义残差网络
def ResNet_Model(input_shape):
	#定义喂入网络的数据维度
	X_input = Input(input_shape)
	
	#stage_1
	X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), 
				kernel_initializer='he_normal', padding='same',
				kernel_regularizer=regularizers.l2(1e-4))(X_input)
	X = BatchNormalization(epsilon=1e-8)(X)
	X = Activation('relu')(X)
	#stage_2
	X = convolution_block(X, (3, 3), [16, 16, 64], s=1, conv_first=False)
	X = identity_block(X, (3, 3), [16, 16, 64])
	X = identity_block(X, (3, 3), [16, 16, 64])
	
	#stage_3
	X = convolution_block(X, (3, 3), [64, 64, 128], s=2)
	X = identity_block(X, (3, 3), [64, 64, 128])
	X = identity_block(X, (3, 3), [64, 64, 128])
	
	# #stage_4
	X = convolution_block(X, (3, 3), [128, 128, 256], s=2)
	X = identity_block(X, (3, 3), [128, 128, 256])
	X = identity_block(X, (3, 3), [128, 128, 256])

	#均值池化
	X = BatchNormalization(epsilon=1e-8)(X)
	X = Activation('relu')(X)
	X = AveragePooling2D((8, 8))(X)

	#输出层
	X = Flatten()(X)
	X = Dropout(0.3)(X)
	X = Dense(10, activation='softmax', kernel_initializer='he_normal')(X)
	#创建模型
	model = Model(inputs=X_input, outputs=X)
	return model

#数据增强及迭代器
def data_gen():
	(X_train_image, y_train_label), (X_test_image, y_test_label) = cifar10.load_data()
	X_train_normalize = X_train_image.astype('float32')/255.
	X_test_normalize = X_test_image.astype('float32')/255.
	y_train_onehot = np_utils.to_categorical(y_train_label)
	y_test_onehot =  np_utils.to_categorical(y_test_label)
	datagen = ImageDataGenerator(
	    	  rotation_range=0,
	          width_shift_range=0.1,
	          height_shift_range=0.1,
	          horizontal_flip=True,
	          vertical_flip=False)
	datagen.fit(X_train_normalize)
	train_data = datagen.flow(X_train_normalize, y_train_onehot, batch_size=32)
	return train_data, X_test_normalize, y_test_onehot

#定义学习率
def lr_schedule(epoch):
	lr = 1e-3
	if epoch > 180:
		lr *= 0.5e-3
	elif epoch > 160:
		lr *= 1e-3
	elif epoch > 120:
		lr *= 1e-2
	elif epoch > 100:
		lr *= 1e-1
	print('learning rate: ', lr)
	return lr 

#定义回调函数
def check_point():

	#保存权重数据
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	model_name = 'cifar10_ResNet_model.{epoch:03d}.h5'
	if not os.path.isdir(save_dir):
	    os.makedirs(save_dir)
	filepath = os.path.join(save_dir, model_name)
	checkpoint = ModelCheckpoint(filepath=filepath,
	                             monitor='val_acc',
	                             verbose=1,
	                             save_best_only=True)
	#学习率调度器
	lr_scheduler = LearningRateScheduler(lr_schedule)

	#当检测值不再上升时，减少学习率
	lr_reducer = ReduceLROnPlateau(factor=0.5,
                               	   cooldown=0,
                               	   patience=5,
                               	   min_lr=0.)

	callbacks = [checkpoint, lr_scheduler, lr_reducer, TensorBoard(log_dir='./tmp/log_ResNet32v2_2')]
	return callbacks

#定义主函数
def main():

	#数据及模型载入
	train_data, X_test_normalize, y_test_onehot = data_gen()
	model = ResNet_Model(input_shape=(32, 32, 3))
	# model.load_weights('saved_models/cifar10_ResNet_model.200.h5')
	model.summary()
	callbacks = check_point()

	#模型优化器/损失函数定义
	model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit_generator(generator=train_data, 
								  epochs=300, verbose=1,
								  callbacks=callbacks, 
								  validation_data=(X_test_normalize, y_test_onehot),
								  initial_epoch=0)

#调用主函数运行程序								  
if __name__ == '__main__':
	main()