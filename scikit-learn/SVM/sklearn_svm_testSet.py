from sklearn import svm
from numpy import *
import matplotlib.pyplot as plt
from pylab import mpl

#中文显示
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

#载入数据
def loadDataSet():
	dataMat = []; labelMat = []
	fr = loadtxt('testSet.txt',  delimiter='\t')
	for i in fr:
		dataMat.append([float(i[0]), float(i[1])])
		labelMat.append(int(i[2]))
	return dataMat, labelMat

def plotLine(dataMat, labelMat):
	x1 = []; y1 = [];
	x2 = []; y2 = []
	for i in range(len(labelMat)):
		if labelMat[i] == 1:
			x1.append(dataMat[i][0])
			y1.append(dataMat[i][1])
		else:
			x2.append(dataMat[i][0])
			y2.append(dataMat[i][1])
	plt.scatter(x1, y1)
	plt.scatter(x2, y2)
	plt.show()

#限定x，y的坐标范围与间隔
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))
    return xx, yy




dataMat, labelMat = loadDataSet()
print(dataMat)
'''
C = 1.0
#无核线性模型 
#clf = svm.LinearSVC(C=C)
#为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）                       
clf = svm.SVC(kernel='linear', C=C)
#为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。           
#clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
#高次函数核  
#clf = svm.SVC(kernel='poly', degree=3, C=C)    
clf.fit(dataMat, labelMat)
plt.subplot(111)
xx, yy = make_meshgrid(array(dataMat)[:,0], array(dataMat)[:,1])
Z = clf.predict(c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plotLine(dataMat, labelMat)
plt.show()
'''