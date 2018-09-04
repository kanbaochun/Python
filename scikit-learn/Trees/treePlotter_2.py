import matplotlib.pyplot as plt
from pylab import mpl

mytree = {'no surfacing': {0: 'no', 1: {'flipers': {0: 'no', 1: 'yes'}}}}

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False
#使用文本注解绘制树节点  
#包含了边框的类型，边框线的粗细等  
decisionnode=dict(boxstyle="sawtooth",fc="0.8",pad=1)# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细  ,pad指的是外边框锯齿形（圆形等）的大小  
leafnode=dict(boxstyle="round4",fc="0.8",pad=1)# 定义决策树的叶子结点的描述属性 round4表示圆形  
arrow_args=dict(arrowstyle="<-")#定义箭头属性  
  
def plotnode(nodetxt,centerpt,parentpt,nodetype):  
    # annotate是关于一个数据点的文本    
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点    
    #annotate的作用是添加注释，nodetxt是注释的内容，  
    #nodetype指的是输入的节点（边框）的形状  
    createplot.ax1.annotate(nodetxt,xy=parentpt,xycoords='axes fraction', 
                           xytext=centerpt,textcoords='axes fraction',  
                           va="center",ha="center",bbox=nodetype,arrowprops=arrow_args)

def getnumleafs(mytree):#计算叶子节点的个数（不包括中间的分支节点）  
    numleafs=0  
  #原代码  firststr=mytree.keys()[0]  # 获得myTree的第一个键值，即第一个特征，分割的标签   
    firststr=list(mytree.keys())[0]  
    #遇到的问题是mytree.keys()获得的类型是dict_keys，而dict_keys不支持索引，我的解决办法是把获得的dict_keys强制转化为list即可  
    seconddict=mytree[firststr]# 根据键值得到对应的值，即根据第一个特征分类的结果    
    for key in seconddict.keys():  #获取第二个小字典中的key  
        if type(seconddict[key]).__name__=='dict':#判断是否小字典中是否还包含新的字典（即新的分支）   书上写的是.-name-但是3.0以后得版本都应该写成.__name__(两个下划线)  
            numleafs +=getnumleafs(seconddict[key])#包含的话进行递归从而继续循环获得新的分支所包含的叶节点的数量  
        else: numleafs +=1#不包含的话就停止迭代并把现在的小字典加一表示这边有一个分支  
    return numleafs  
  
  
def gettreedepth(mytree):#计算判断节点的个数  
    maxdepth=0  
    firststr=list(mytree.keys())[0]  
    seconddict=mytree[firststr]  
    for key in seconddict.keys():  
        if type(seconddict[key]).__name__=='dict':  
            thisdepth  = 1+gettreedepth(seconddict[key])  
        else: thisdepth =1  
        if thisdepth>maxdepth:  
            maxdepth=thisdepth#间隔 间隔间隔得问题一定要多考虑啊啊啊啊啊啊  
    return maxdepth
def plotmidtext(cntrpt,parentpt,txtstring):#作用是计算tree的中间位置    cntrpt起始位置,parentpt终止位置,txtstring：文本标签信息  
    xmid=(parentpt[0]-cntrpt[0])/2.0+cntrpt[0]# cntrPt 起点坐标 子节点坐标   parentPt 结束坐标 父节点坐标  
    ymid=(parentpt[1]-cntrpt[1])/2.0+cntrpt[1]#找到x和y的中间位置  
    createplot.ax1.text(xmid,ymid,txtstring)  
      
      
def plottree(mytree,parentpt,nodetxt):  
    numleafs=getnumleafs(mytree)  
    depth=gettreedepth(mytree)  
    firststr=list(mytree.keys())[0]  
    cntrpt=(plottree.xoff+(1.0+float(numleafs))/2.0/plottree.totalw,plottree.yoff)#计算子节点的坐标   
    plotmidtext(cntrpt,parentpt,nodetxt) #绘制线上的文字    
    plotnode(firststr,cntrpt,parentpt,decisionnode)#绘制节点    
    seconddict=mytree[firststr]  
    plottree.yoff=plottree.yoff-1.0/plottree.totald#每绘制一次图，将y的坐标减少1.0/plottree.totald，间接保证y坐标上深度的  
    for key in seconddict.keys():  
        if type(seconddict[key]).__name__=='dict':  
            plottree(seconddict[key],cntrpt,str(key))  
        else:  
            plottree.xoff=plottree.xoff+1.0/plottree.totalw  
            plotnode(seconddict[key],(plottree.xoff,plottree.yoff),cntrpt,leafnode)  
            plotmidtext((plottree.xoff,plottree.yoff),cntrpt,str(key))  
    plottree.yoff=plottree.yoff+1.0/plottree.totald  
  
      
def createplot(intree):  
     # 类似于Matlab的figure，定义一个画布(暂且这么称呼吧)，背景为白色   
    fig=plt.figure(1,facecolor='white')  
    fig.clf()    # 把画布清空   
    axprops=dict(xticks=[],yticks=[])     
    # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图   
    # frameon表示是否绘制坐标轴矩形   
    createplot.ax1=plt.subplot(111,frameon=False,**axprops)   
    plottree.totalw=float(getnumleafs(intree))  
    plottree.totald=float(gettreedepth(intree))  
    plottree.xoff=-0.6/plottree.totalw;plottree.yoff=1.1;  
    plottree(intree,(1.0,2.0),'')  
    plt.show()

createplot(mytree)
