from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add
from keras.layers import AveragePooling2D, Input, Flatten, ZeroPadding2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os

#定义残差块(skip 3 layers)
def identity_block(X, kernel_size, filters):
	X_shortcut = X
	f1, f2, f3 = filters
	#改变通道数量
	X = Conv2D(filters=f1, kernel_size=(1, 1), 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal')(X)
	X = BatchNormalization(axis = 3, epsilon=1e-8)(X)
	X = Activation('relu')(X)
	#卷积提取信息操作
	X = Conv2D(filters=f2, kernel_size=kernel_size, 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal',padding='same')(X)
	X = BatchNormalization(axis = 3, epsilon=1e-8)(X)
	X = Activation('relu')(X)
	#改变通道数量
	X = Conv2D(filters=f3, kernel_size=(1, 1), 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal',padding='same')(X)
	X = BatchNormalization(axis = 3, epsilon=1e-8)(X)
	#建立shortcut(添加残差块到主路径)
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)
	return X

#定义卷积块(skip 3 layers)
def convolution_block(X, kernel_size, filters, s=2):
	X_shortcut = X
	f1, f2, f3 = filters
	#改变通道数量
	X = Conv2D(filters=f1, kernel_size=(1, 1), 
				strides=(s, s), activation='relu', 
				kernel_initializer='he_normal')(X)
	X = BatchNormalization(axis = 3, epsilon=1e-8)(X)
	X = Activation('relu')(X)
	#卷积操作提取信息
	X = Conv2D(filters=f2, kernel_size=kernel_size, 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal',padding='same')(X)
	X = BatchNormalization(axis = 3, epsilon=1e-8)(X)
	X = Activation('relu')(X)
	##改变通道数量
	X = Conv2D(filters=f3, kernel_size=(1, 1), 
				strides=(1, 1), activation='relu', 
				kernel_initializer='he_normal',padding='same')(X)
	X = BatchNormalization(axis = 3, epsilon=1e-8)(X)
	#shortcut path上的卷积操作
	X_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), 
				strides=(s, s), activation='relu', 
				kernel_initializer='he_normal',padding='same')(X_shortcut)
	X_shortcut = BatchNormalization(axis = 3, epsilon=1e-8)(X_shortcut)
	#建立shortcut(添加残差块到主路径)
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)
	return X

#定义残差网络
def resnet_v20(input_shape):
	#定义喂入网络的数据维度
	X_input = Input(input_shape)
	
	#stage_1
	X = ZeroPadding2D((3, 3))(X_input)
	X = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), 
				kernel_initializer='he_normal', padding='valid')(X)
	X = BatchNormalization(axis = 3, epsilon=1e-8)(X)
	#stage_2
	X = convolution_block(X, (3, 3), [32, 32, 64], s=1)
	X = identity_block(X, (3, 3), [32, 32, 64])
	
	#stage_3
	X = convolution_block(X, (3, 3), [64, 64, 128], s=2)
	X = identity_block(X, (3, 3), [64, 64, 128])
	X = identity_block(X, (3, 3), [64, 64, 128])
	
	#stage_4
	X = convolution_block(X, (3, 3), [128, 128, 128], s=2)
	X = identity_block(X, (3, 3), [128, 128, 128])
	X = identity_block(X, (3, 3), [128, 128, 128])
	X = identity_block(X, (3, 3), [128, 128, 128])

	#均值池化
	X = AveragePooling2D((7, 7))(X)
	#输出层
	X = Flatten()(X)
	X = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(X)
	#创建模型
	model = Model(inputs=X_input, outputs=X)
	return model

#数据增强及迭代器
def data_gen(train_path, test_path):
	datagen = ImageDataGenerator(
	    rescale=1./255,
	    rotation_range=0,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    horizontal_flip=True)
	train_data = datagen.flow_from_directory(train_path,
											 target_size = (224, 224),
											 batch_size=32,
											 class_mode='binary')
	validation_data = datagen.flow_from_directory(test_path,
												  target_size = (224, 224),
												  batch_size=32,
												  class_mode='binary')
	return train_data, validation_data

#定义回调函数
def check_point():
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	model_name = 'kaggle_20v1_model.{epoch:03d}.h5'
	if not os.path.isdir(save_dir):
	    os.makedirs(save_dir)
	filepath = os.path.join(save_dir, model_name)
	#定义回调函数
	checkpoint = ModelCheckpoint(filepath=filepath,
	                             monitor='val_acc',
	                             verbose=1,
	                             save_best_only=True)

	callbacks = [checkpoint]
	return callbacks

#定义主函数
def main():

	#数据及模型载入
	train_path = 'train'
	test_path = 'test2'
	train_data, validation_data = data_gen(train_path, test_path)
	# try:
	#     model = load_model('saved_models/kaggle_20v1_model.087.h5')
	#     print('模型加载成功！继续训练模型')
	# except:
	# 	print('模型加载失败，重新训练模型')
	model = resnet_v20(input_shape = (224, 224, 3))
	model.summary()
	callbacks = check_point()

	#模型优化器/损失函数定义
	model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
	history = model.fit_generator(generator=train_data, 
								  epochs=100, verbose=1,
								  callbacks=callbacks, 
								  validation_data=validation_data)
								  

if __name__ == '__main__':
	main()