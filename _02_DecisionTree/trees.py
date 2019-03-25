'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator
import treePlotter


# 创建决策树使用的数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    # change to discrete values
    return dataSet, labels


# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # 计算每个标签的出现频次
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)  # 以2为底的对数，香农熵计算公式
    return shannonEnt


# 决策树中常见，划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # （1）截取划分点前面的数据
            reducedFeatVec.extend(featVec[axis+1:])  # （2）截取划分点后面的数据
            retDataSet.append(reducedFeatVec)  # 将划分点前后的数据进行存储
    return retDataSet


# 决策树寻找最优划分点（暴力循环计算并查找）
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数（最后一列是标签，不是特征）
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集合的 基准香农熵
    bestInfoGain = 0.0
    bestFeature = -1

    # 遍历所有的特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 获取该特征下的所有特征值
        uniqueVals = set(featList)  # 获取该特征值集合（不重复）
        newEntropy = 0.0
        # 计算该特征值划分数据后的香农熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 注意信息增益公式
        infoGain = baseEntropy - newEntropy  # 计算熵的差值（信息增益，正值）
        if (infoGain > bestInfoGain):  # 比较得到最大的信息增益
            bestInfoGain = infoGain
            bestFeature = i  # 数据集的特征划分点
    return bestFeature  # 返回最优的特征划分点


# 没有特征可以划分的情况下，利用投票原则选择出该节点中类别最多的样本 作为该节点的类别信息
def majorityCnt(classList):
    classCount = {}  # 类别计数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 按照value排序（降序）
    return sortedClassCount[0][0]  # 返回最多的类别


# 递归建立决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 获取标签数组
    if classList.count(classList[0]) == len(classList):  # 递归结束条件1：一个节点的所有类别都相同，无需继续分割
        return classList[0]
    if len(dataSet[0]) == 1:  # 递归结束条件2：数据集中没有其他特征可以使用，都已经遍历过了
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])  # 删除对应的特征，下次不会再对其进行考虑
    featValues = [example[bestFeat] for example in dataSet]  # 数据集中所有的特征值列表
    uniqueVals = set(featValues)  # 数据集中所有的特征值集合，集合大小对应树上的节点个数
    for value in uniqueVals:  # 决策树的一个节点
        subLabels = labels[:]  # 复制所有标签，决策树不会被已使用过的标签混乱
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree                            


# 真正使用决策树，完成分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]  # 列别标签
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 列别标签转换为列别索引（字符串转数值型）
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)  # 走到了另一个分支节点，继续分类
    else:
        classLabel = valueOfFeat  # 走到了叶子节点，返回对应标签
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':

    # 1、计算香农熵
    my_dataSet, labels = createDataSet()
    print(len(my_dataSet), len(labels))
    shannonEnt_1 = calcShannonEnt(my_dataSet)
    my_dataSet[0][-1] = 'maybe'  # 改变数据集，重新计算香农熵
    shannonEnt_2 = calcShannonEnt(my_dataSet)
    print('shannonEnt_1 【%f】 and shannonEnt_2 【%f】' %(shannonEnt_1,shannonEnt_2))  # 会发现类别越多，熵越大

    # 2、按照给定特征划分数据集合
    dataSet_1, labels = createDataSet()
    print(dataSet_1)
    split_dataSet = splitDataSet(dataSet_1, 0, 0)
    print(split_dataSet)

    # 3、不断尝试各个数据划分点，找到熵最小的最优划分点
    best_split_point = chooseBestFeatureToSplit(dataSet_1)
    print('best_split_point:【%d】' % best_split_point)

    # 4、建立决策树
    decision_tree = createTree(dataSet_1, labels)
    print(decision_tree)

    # 5、使用决策树，完成分类
    dataSet_1, labels = createDataSet()
    rs_label = classify(decision_tree, labels, [1, 0])
    print('result label 【%s】' % rs_label)

    # 6、存储和加载模型
    model_file = './tree_model.pkl'
    storeTree(decision_tree, model_file)
    decision_tree_1 = grabTree(model_file)
    print(decision_tree_1)

    # 7、真实数据场景，使用决策树分类器
    fr = open('./lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_tree = createTree(lenses, lenses_labels)
    print(lenses_tree)
    treePlotter.createPlot(lenses_tree)
