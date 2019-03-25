'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''

from numpy import *
import numpy as np

# 加载数据
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('../testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid函数（非线性），输出为概率值大小
def sigmoid(inX):
    return 1.0/(1+exp(-inX))


# 梯度下降（上升）
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)   # 转化为NumPy matrix
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)  # 样本数m，特征数n
    m1, n1 = shape(labelMat)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))  # 权重初始化
    for k in range(maxCycles):              # heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     # matrix mult（矩阵相乘运算，得到每个样本的概率）
        error = (labelMat - h)              # vector subtraction（向量相减，得到每个样本的误差）
        # 下面的公式是对均方误差（交叉熵）求导后的结果
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult（矩阵更新）
    return weights

# 画出 数据集合 和 逻辑回归的最佳拟合直线函数（决策边界）
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]  # n代表样本数量
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:  # 如果样本实际类别为1
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])  # 将2维的特征进行可视化
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)  # x取值
    y = (-weights[0]-weights[1]*x)/weights[2]  # 根据参数和x，得到y值（得到的方程可以推导得到，x0是1）
    ax.plot(x, y)  # 画出决策边界
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 第一版 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # 初始化参数为1
    for i in range(m):
        inX = dataMatrix[i] * weights
        h = sigmoid(sum(inX))  # 这里的h是个数值
        error = classLabels[i] - h
        ins = dataMatrix[i] * np.array(alpha) * np.array(error)
        weights = weights + ins
    return weights


# 改进版本 随机梯度上升（）
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)   # 初始化参数为1
    for j in range(numIter):  # 迭代次数j
        dataIndex = list(range(m))
        for i in range(m):  # 样本编号i
            alpha = 4/(1.0+j+i)+0.0001  # 步长alpha随着迭代次数的增加，逐渐减小（由于常数项存在，不会减小到0）
            randIndex = int(random.uniform(0, len(dataIndex)))  # 获取随机样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + dataMatrix[randIndex] * np.array(alpha) * np.array(error)  # 梯度上升公式
            del(dataIndex[randIndex])
    return weights

# 利用学得参数，进行分类
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

# 使用实际数据 测试逻辑回归
def colicTest():
    # 整理训练和测试数据
    frTrain = open('./horseColicTraining.txt')
    frTest = open('./horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):  # 20维特征，1个label
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    # 训练过程
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):  # 判断分类正确性
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


if __name__ == '__main__':

    # 1、加载数据，梯度下降
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    print(type(weights), shape(weights))

    # 2、画出决策边界
    plotBestFit(weights.getA())

    # 3、随机梯度上升（效果不理想，错分挺多）
    weights0 = stocGradAscent0(dataMat, labelMat)
    plotBestFit(weights0)

    # 4、改进版本梯度上升
    weights1 = stocGradAscent1(dataMat, labelMat)
    plotBestFit(weights1)

    # 5、实际数据测试逻辑回归
    colicTest()