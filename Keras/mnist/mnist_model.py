from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import json
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import LearningRateScheduler

#绘制训练过程图像
def show_train_history(train_history, train, validation):
	plt.plot(train_history.history[train])
	plt.plot(train_history.history[validation])
	plt.title('Train history')
	plt.ylabel('train')
	plt.xlabel('Epoch')
	#设置图例
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

#显示预测结果
def plot_image(image, label, prediction, idx, num):
	fig = plt.gcf()
	fig.set_size_inches(6, 10)
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

#数据集载入及预处理
(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
X_train = X_train_image.reshape(60000, 784).astype('float32')
X_test = X_test_image.reshape(10000, 784).astype('float32')
X_train_normalize = X_train/255
X_test_normalize = X_test/255
y_train_onehot = np_utils.to_categorical(y_train_label)
y_test_onehot =  np_utils.to_categorical(y_test_label)

#神经网络模型建立(2个隐藏层)
model = Sequential()
model.add(Dense(units=1000, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
print(model.summary())

#学习率调整
def lr_schedule(epoch):
	lr = 1e-3
	if epoch >= 20:
		lr *= 1e-1
	elif epoch >= 10:
		lr *= 1e-1
	print('learning rate: ', lr)
	return lr
lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks=[lr_scheduler]
#训练模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
train_history = model.fit(x=X_train_normalize, y=y_train_onehot, validation_split=0.2, 
							epochs=30, batch_size=200, verbose=2, callbacks=callbacks)

#评估模型的准确率
scores = model.evaluate(X_test_normalize, y_test_onehot)
print()
print('accuracy= ', scores[1])
prediction = model.predict_classes(X_test)
#plot_image(X_test_image, y_test_label, prediction, idx=340, num=10)

#显示混淆矩阵
#pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])

#保存训练信息
with open('model/model.json', 'w') as f:
	json.dump(train_history.history, f)
	print('保存成功')



	




