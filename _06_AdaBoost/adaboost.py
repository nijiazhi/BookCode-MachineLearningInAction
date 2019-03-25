'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

# 加载简单数据
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


# 加载文件中数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))  # 获取特征数量
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


'''
    通过阈值比较对数据进行分类（使用数组过滤的方式实现）
    dimen      ： 对应特征维度
    threshVal  ： 阈值的数值
    threshIneq ： 阈值的判断方向（大于小于）
'''
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))  # 建立全部为1的数组
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0  # 数组过滤
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

'''
    遍历所有特征，找到最佳单层决策树
    D ： 权重向量，初始化为：1/样本数m
'''
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)

    numSteps = 10.0  # 在特征的所有可能值上遍历，对应步数
    bestStump = {}  # 字典存放【给定权重向量D：对应的最佳单层决策树】
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # 初始化最小误差，正无穷

    for i in range(n):  # 遍历所有特征
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()  # 利用最大最小值来计算步长
        stepSize = (rangeMax-rangeMin) / numSteps  # 对应的步长

        # 遍历该特征上所有的取值
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:  # 遍历大于和小于情况
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 调用分类函数
                errArr = mat( ones((m,1)) )
                errArr[predictedVals == labelMat] = 0  # 构建误差向量
                weightedError = D.T * errArr  # 误差向量乘上权重向量（计算加权错误率）
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"
                      % (i, threshVal, inequal, weightedError))

                if weightedError < minError:  # 满足最优条件，进行更新
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


'''
    完整版adaboost算法
    numIt ： 最大迭代轮数
'''
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []  # 存放所有弱分类器的列表
    m = shape(dataArr)[0]
    D = mat( ones((m,1))/m )  # 初始化样本权重向量
    aggClassEst = mat( zeros((m,1)) )  # 类别估计累计值

    for i in range(numIt):
        # 获取最好的分类树桩（当前的弱分类器），error是当前的分类错误率
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("权重向量D:", D.T)
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))  # 计算alpha值（分类器的权重）
        bestStump['alpha'] = alpha  # 记录当前alpha
        weakClassArr.append(bestStump)  # 把当前最好的分类树桩，存储到列别中
        print("分类结果classEst: ", classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)  # 为下一次迭代计算新的样本权重向量D
        D = multiply(D, exp(expon))
        D = D/D.sum()

        # 计算所有分类器累加的训练错误率，如果为0直接跳出循环
        aggClassEst += alpha*classEst
        print("累加的类别估计值aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("累加的训练错误率total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


'''
    adaboost分类函数
    datToClass    ： 待分类样本
    classifierArr ： 已经训练得到的弱分类器和对应权重
'''
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]  # 样本数量
    aggClassEst = mat(zeros((m, 1)))  # 累计的分类结果
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])  # 调用决策树桩（弱分类器）进行分类
        aggClassEst += classifierArr[i]['alpha'] * classEst  # 分类结果*对应权重
        # print('当前分类结果:\t', aggClassEst)
    return sign(aggClassEst)


'''
    绘制分类器的Roc曲线
'''
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    print(predStrengths.shape)
    cur = (1.0, 1.0)  # 绘制光标的位置
    ySum = 0.0  # 计算AUC面积需要用到的y轴累积值
    numPosClas = sum(array(classLabels) == 1.0)  # 类别为1的数量
    yStep = 1/float(numPosClas)  # y轴的步长，能把类别为1的样本清空
    xStep = 1/float(len(classLabels) - numPosClas)  # x轴步长

    sortedIndicies = predStrengths.argsort()  # 预测强度排序之后的索引序列（升序）
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    # 遍历所有值，在每个点之间连线
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:  # 类别为1，y轴下降
            delX = 0
            delY = yStep
        else:  # 类别不为1，x轴向左
            delX = xStep
            delY = 0
            ySum += cur[1]  # 记录AUC计算需要的小矩形高度
        # draw line from cur to (cur[0]-delX, cur[1]-delY)
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("xStep: ", xStep)
    print("the Area Under the Curve is: ", ySum*xStep)


if __name__ == '__main__':

    # 1、加载数据，简单版本决策树桩
    datMat, classLabels = loadSimpData()
    D = mat( ones((len(classLabels), 1))/len(classLabels) )  # 初始化的样本权重向量
    bestStump, minError, bestClasEst = buildStump(datMat, classLabels, D)
    print('\nrs1', '\n', bestClasEst, '\n', bestStump,'\n', minError, '\n\n')

    # 2、完整版本adaboost算法
    weakClassArr, aggClassEst = adaBoostTrainDS(datMat, classLabels, 9)  # 输出所有的最优树桩和最终类别的累计值
    print('\nrs2', '\n', weakClassArr, '\n', aggClassEst, '\n\n')
    classify_rs = adaClassify([0, 0], weakClassArr)
    print(classify_rs)

    # 3、使用adaboost预测真实数据集（和logistic做比较）
    dataMat_train, labelMat_train = loadDataSet('./horseColicTraining2.txt')
    weakClassArr1, aggClassEst1 = adaBoostTrainDS(dataMat_train, labelMat_train, 50)
    dataMat_test, labelMat_test = loadDataSet('./horseColicTest2.txt')
    predict_10 = adaClassify(dataMat_test, weakClassArr1)
    errArr = mat(ones((67, 1)))
    err_number = errArr[predict_10 != mat(labelMat_test).T].sum()
    print('错误率：', err_number/errArr.shape[0])

    # 4、画出adaboost的Roc曲线
    dataMat_train, labelMat_train = loadDataSet('./horseColicTraining2.txt')
    weakClassArr1, aggClassEst1 = adaBoostTrainDS(dataMat_train, labelMat_train, 50)
    print(aggClassEst1.shape)
    plotROC(aggClassEst1.T, labelMat_train)


