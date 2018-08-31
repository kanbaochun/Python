from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.optimizers import Adam
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.utils import np_utils
import os

#卷积块
def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    x = Activation('relu')(input)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    return x

#过渡块
def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
	'''
	concat_axis通道顺序设置:
   	'th'表示'threano'后端，对应(channels, height, width)，
   	'tf'表示'tensorflow'后端，对应(height, width, channels)
   	'''
	concat_axis = 1 if K.image_dim_ordering() == "th" else -1
	x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(input)
	if dropout_rate is not None:
		x = Dropout(dropout_rate)(x)
	x = AveragePooling2D((2, 2), strides=(2, 2))(x)
	x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
	return x

#DenseNet块
def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
	'''
	concat_axis通道顺序设置:
   	'th'表示'threano'后端，对应(channels, height, width)，
   	'tf'表示'tensorflow'后端，对应(height, width, channels)
   	'''
	concat_axis = 1 if K.image_dim_ordering() == "th" else -1
	feature_list = [x]
	for i in range(nb_layers):
		x = conv_block(x, growth_rate, dropout_rate, weight_decay)
		feature_list.append(x)
		x = Concatenate(axis=concat_axis)(feature_list)
		nb_filter += growth_rate
	return x, nb_filter

#创建DenseNet网络
def DenseNet_model(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                     weight_decay=1E-4, verbose=True):

    model_input = Input(shape=img_dim)
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    assert (depth - 4) % nb_dense_block == 0, "Depth must be 3 N + 4"

    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(model_input)
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                            beta_regularizer=l2(weight_decay))(x)

    #每个dense_block的层数,nb_dense_block设置多少个dense_block
    nb_layers = int((depth - 4) / nb_dense_block)
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
  
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=model_input, outputs=x)

    if verbose: 
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet

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
	if epoch >= 200:
		lr *= 1e-3
	elif epoch >= 150:
		lr *= 1e-2
	elif epoch >= 100:
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
                               	   monitor='val_acc',
                               	   patience=5,
                               	   min_lr=0.)

	callbacks = [checkpoint, lr_scheduler, lr_reducer, TensorBoard(log_dir='./tmp/log_DenseNet_2')]
	return callbacks

#定义主函数
def main():

	#数据及模型载入
	train_data, X_test_normalize, y_test_onehot = data_gen()
	model = DenseNet_model(10, img_dim=(32, 32, 3), dropout_rate=0.2)
	#model.load_weights('saved_models/cifar10_ResNet_model.018.h5')
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