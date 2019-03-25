'''
Created on Oct 6, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logRegres

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.5
    weights = ones(n)   #initialize to all ones
    weightsHistory=zeros((500*m,n))
    for j in range(500):
        for i in range(m):
            h = logRegres.sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
            weightsHistory[j*m + i,:] = weights
    return weightsHistory


# 梯度上升过程中，记录权重的变化情况
def stocGradAscent1(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.4
    weights = ones(n)   # initialize to all ones
    weightsHistory = zeros((40*m, n))
    for j in range(40):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = logRegres.sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]  # 三个维度的权重
            weightsHistory[j*m + i, :] = weights  # 记录每一次的权重变化
            del(dataIndex[randIndex])
    print(weights)
    return weightsHistory
    
# 加载数据
dataMat, labelMat = logRegres.loadDataSet()
dataArr = array(dataMat)
myHist = stocGradAscent1(dataArr, labelMat)  # 权重变化历史


n = shape(dataArr)[0]  # number of points to create
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []

# 点的标识 和 颜色
markers = []
colors = []

fig = plt.figure()

ax = fig.add_subplot(311)  # 图像分成3部分，在第三位的1 2 3，表示垂直分成了三份
type1 = ax.plot(myHist[:, 0])
plt.ylabel('X0')

ax = fig.add_subplot(312)
type1 = ax.plot(myHist[:, 1])
plt.ylabel('X1')

ax = fig.add_subplot(313)
type1 = ax.plot(myHist[:, 2])
plt.xlabel('iteration')
plt.ylabel('X2')

plt.show()