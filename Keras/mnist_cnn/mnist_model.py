from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import json
import matplotlib.pyplot as plt
import pandas as pd 

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
	fig.set_size_inches(10, 6)
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
X_train = X_train_image.reshape(X_train_image.shape[0], 28, 28, 1).astype('float32')
X_test = X_test_image.reshape(X_test_image.shape[0], 28, 28, 1).astype('float32')
X_train_normalize = X_train/255
X_test_normalize = X_test/255
y_train_onehot = np_utils.to_categorical(y_train_label)
y_test_onehot =  np_utils.to_categorical(y_test_label)

#卷积神经网络模型建立
model = Sequential()
#卷积与池化层1
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#卷积与池化层2
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#平坦层(单个训练数据转换成一维)
model.add(Flatten())
#建立全连接(1个隐藏层)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.summary())

#训练模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=X_train_normalize, y=y_train_onehot, validation_split=0.2, 
							epochs=10, batch_size=300, verbose=2)

#评估模型的准确率
scores = model.evaluate(X_test_normalize, y_test_onehot)
print()
print('accuracy= ', scores[1])
prediction = model.predict_classes(X_test_image)
print(prediction[:10])


#显示混淆矩阵
pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])

#保存训练信息
with open('model/model.json', 'w') as f:
	json.dump(train_history.history, f)
	print('保存成功')



	




