'''
Created on Oct 14, 2010

@author: Peter Harrington
'''

import matplotlib.pyplot as plt
import trees

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 分支节点
leafNode = dict(boxstyle="round4", fc="0.8")  # 叶子节点
arrow_args = dict(arrowstyle="<-")

# 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 最初级的画图函数
def createPlot_1():
   fig = plt.figure(1, facecolor='white')
   fig.clf()
   createPlot.ax1 = plt.subplot(111, frameon=False)  # demo演示使用
   plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
   plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
   plt.show()


# 返回决策树的叶子节点个数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 判断节点类型（字典型的分支节点 或者 叶子节点）
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 返回决策树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 判断节点类型（字典型的分支节点 或者 叶子节点）
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 在父子节点间填充文本信息（分支的具体内容）
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]  # 在父节点和子节点之间，计算中心位置
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 画决策树的子函数
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  # 树的宽度
    depth = getTreeDepth(myTree)  # 树的高度
    firstStr = list(myTree.keys())[0]  # 这个节点的文本内容
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  # totalD和totalW都在0-1的变化范围内
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))  # 继续递归
        else:  # 如果是叶子节点，直接画出该节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()  # 清空画板
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False)  # ticks for demo purposes
    plotTree.totalW = float(getNumLeafs(inTree))  # 宽度（全局变量）
    plotTree.totalD = float(getTreeDepth(inTree))  # 高度
    plotTree.xOff = -0.5/plotTree.totalW  # 追踪已经绘制节点位置 和 放置下个节点的恰当位置
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# 预先存储好的决策树字典
def retrieveTree(i):
    listOfTrees =\
    [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


if __name__ == '__main__':

    # 1、构建决策树
    dataSet, labels = trees.createDataSet()
    thisTree = trees.createTree(dataSet, labels)
    print(thisTree)

    # 2、简单画图
    createPlot_1()

    # 3、获取树叶子节点个数和深度
    leaf_number = getNumLeafs(thisTree)
    depth = getTreeDepth(thisTree)
    print('leftNumber:【%f】, depth:【%f】' %(leaf_number, depth))

    # 4、画出整棵树
    tree = retrieveTree(0)
    print(tree)
    createPlot(tree)
