from math import log
import operator

#建立数据集
def createDataSet():
    DataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing', 'flipers'] 
    return DataSet, labels

#计算给定数据集的香农熵
def calcShannonEnt(DataSet):
    numEntries=len(DataSet)
    labelCounts={}
    for featVec in DataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
        shannoEnt=0.0
        for key in labelCounts:
            prob=float(labelCounts[key]/numEntries)
            shannoEnt-=prob*log(prob,2)
    return shannoEnt

#选择最好的数据集划分方式(熵最大)
def chooseBestFeatureToSplit(DataSet):
    numFeatures = len(DataSet[0]) - 1     
    baseEntropy = calcShannonEnt(DataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):                        
        featList = [example[i] for example in DataSet]  
        uniqueVals = set(featList)                      
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(DataSet, i, value)
            prob = len(subDataSet)/float(len(DataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy             
        if (infoGain > bestInfoGain):                   
            bestInfoGain = infoGain                    
            bestFeature = i
    return bestFeature

#按照给定特征划分数据集,筛选axis轴值为Value数据列表，返回符合value值且删除该轴后的列表
def splitDataSet(DataSet, axis, value):   
    retDataSet = []                        
    for featVec in DataSet:                
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]  
            reduceFeatVec.extend(featVec[axis+1:]) #extend拓展列表
            retDataSet.append(reduceFeatVec)       #append添加元素
    return retDataSet

#寻找具有最大类别数量的叶节点
def majoritycnt(classlist):  
    classcount={}  
    for vote in classlist:  
        if vote not in classcount.keys():  
            classcount[vote]=0    
            classcount[vote] +=1  
        sortedclasscount=sorted(classcount.items(),
            key=operator.itemgetter(1),reverse=True)
        return sortedclasscount[0][0]

#创建树的函数代码
def createtree(DataSet,labels):  
    classlist=[example[-1] for example in DataSet]    
    if classlist.count(classlist[0])==len(classlist):  
        return classlist[0]  
    if len(DataSet[0])==1:    
        return majoritycnt(classlist)  
    bestfeat=chooseBestFeatureToSplit(DataSet)  
    bestfeatlabel=labels[bestfeat] 
    mytree={bestfeatlabel:{}}   
    del(labels[bestfeat])   
    featvalues=[example[bestfeat] for example in DataSet]   
    uniquevals=set(featvalues)     
    for value in uniquevals:  
        sublabels=labels[:]    
        mytree[bestfeatlabel][value]=createtree(splitDataSet(DataSet,bestfeat,value),sublabels) 
    return mytree

#创建数据集
DataSet,labels = createDataSet()
print('训练数据集和标签：\n', DataSet, labels)
#计算熵
shang = calcShannonEnt(DataSet)
print('数据集熵最大的特征：\n', shang)
#选择最好的数据划分方式
choose = chooseBestFeatureToSplit(DataSet)
print('数据集最好的划分特征：\n', choose)
#划分数据集
axis = choose
value = 1
split = splitDataSet(DataSet, axis, value)
print('划分后的数据集：\n', split)
#创建树的函数代码
tree = createtree(DataSet,labels)
print('决策树：\n', tree)
    
