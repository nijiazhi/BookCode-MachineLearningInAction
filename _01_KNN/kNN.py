'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1 x N)
            dataSet: size m data set of known vectors (M x N), M是样本数量，N是特征数量
            labels: data set labels (1 x M vector)
            k: number of neighbors to use for comparison (should be an odd number 奇数)
            
Output:     the most popular class label

@author: pbharrin
'''

from numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # tile是将inX进行复制扩充；相减得到与测试集和训练集的差值
    sqDiffMat = diffMat**2  # 每行相减得到差值后，平方运算（欧式距离运算吧？）
    sqDistances = sqDiffMat.sum(axis=1)  # 每行（各个特征维度上）的差值平方加和（得到了 测试集 和 训练集中每个样本 的欧式距离）
    distances = sqDistances**0.5  # 开方运算，得到真正距离
    sortedDistIndicies = distances.argsort()  # 距离的数组排序，得到排序后的索引序列
    classCount={}  # 字典
    for i in range(k):  # 找到最相邻的k个邻居
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 根据knn投票结果得到答案， 注意sorted可以对字典的key和value分别排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 获取输入文件的行数
    returnMat = zeros((numberOfLines, 3))  # 创建对应维度的矩阵
    classLabelVector = []  # 创建标签向量

    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  # 去除左右空格？或者是‘\n’换行符？
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 赋值每一行
        classLabelVector.append(int(listFromLine[-1]))  # 最后一个数赋值给标签
        index += 1
    return returnMat, classLabelVector


# 对特征数据，进行归一化处理
def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals  # 变化范围
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))  # 用原数据减去最小值构成的矩阵，使最小值变为0（一个好基准）
    normDataSet = normDataSet / tile(ranges, (m,1))  # 除以ranges得到归一化结果（最小值为0，最大值为1）
    return normDataSet, ranges, minVals


# 约会数据的测试程序
def datingClassTest(file_path):
    hoRatio = 0.50  # 测试数据比例
    datingDataMat,datingLabels = file2matrix(file_path)  # 加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化处理
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)  # 测试数据数量
    errorCount = 0.0
    for i in range(numTestVecs):  # 对每一个样本进行knn分类
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)


# 将32*32的图片 转化为 1*1024 的向量
def img2vector(filename):
    returnVect = zeros((1, 1024))  # 创建1*1024向量
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest(data_file_path):
    hwLabels = []
    trainingFileList = listdir(data_file_path)  # 加载手写图片-训练集
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 获取文件名称，当作标签
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # 加载手写图片-测试集
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 获取文件名称，当作标签
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0
    print("\nthe total number of errors is: %d/%d" % (errorCount, mTest))
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':

    # 1、测试knn算法
    group, labels = createDataSet()
    print(group.shape, len(labels))
    classify_result = classify0([0, 0], group, labels, 3)

    # 2、读取文件，转matrix
    data_file = './data/datingTestSet2.txt'
    returnMat, classLabelVector = file2matrix(data_file)
    print(returnMat.shape, len(classLabelVector))

    # 3、归一化操作
    normDataSet, ranges, minVals = autoNorm(returnMat)

    # 4、约会数据knn测试
    datingClassTest(data_file)

    # 5、手写识别数据knn测试
    handwriting_file = './trainingDigits'
    handwritingClassTest(handwriting_file)

