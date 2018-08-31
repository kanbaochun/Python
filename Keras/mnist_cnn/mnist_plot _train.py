import matplotlib.pyplot as plt
import json
import threading

#绘制训练信息图像
def show_train_history(train_history, train, validation):
	plt.plot(train_history[train])
	plt.plot(train_history[validation])
	plt.title('Train history')
	plt.ylabel(train)
	plt.xlabel('Epoch')
	#设置图例
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

with open('model/model.json', 'r') as f:
	train_history = json.load(f)

#绘制准确率图像
show_train_history(train_history, 'acc', 'val_acc')
#绘制损失函数图像
show_train_history(train_history, 'loss', 'val_loss')



