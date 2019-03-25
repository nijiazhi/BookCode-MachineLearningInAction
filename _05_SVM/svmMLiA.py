'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep

# 加载数据集
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


'''
    在0-m的区间范围内，随机选择一个数
    i : 第一个alpha的下标
    m : 所有alpha的数目
'''
def selectJrand(i, m):
    j = i  # 希望选择得到不等于i的数字
    while j == i:
        j = int(random.uniform(0,m))
    return j

# 用于调整大于H或者小于L的alpha值
def clipAlpha(aj, H, L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

'''
    简化版本的SMO算法
    dataMatIn   ： 数据集合
    classLabels ： 类别标签
    C           ： 常数C（松弛变量？）
    toler       ： 容错率
    maxIter     ： 退出前最大循环次数
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)              # 数据矩阵（均转为numpy，便于计算）
    labelMat = mat(classLabels).transpose()  # 类别标签是列向量
    m, n = shape(dataMatrix)  # m是样本数量，n是特征数量
    b = 0
    alphas = mat(zeros((m, 1)))  # alpha列向量，m行1列的全0矩阵初始化

    # iter存放没有alpha改变情况下，遍历数据集的次数。当该值达到maxIter时，函数结束运行退出
    iter = 0

    while iter < maxIter:
        alphaPairsChanged = 0  # 初始化为0，用于记录alpha是否已经进行了优化（循环结束时也同样可以得知这一点）
        for i in range(m):
            # 计算出预测的类别（根据alpha可以根据公式计算w值，另外b=0，利用 f(x)=WX+b 得到分类结果）
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])  # 基于该样本的预测结果和真实结果得到误差Ei（用于检验该样本是否 violates KKT conditions）

            '''
                如果误差Ei很大，则对该实例对应的alpha进行优化
                if语句中对正负间隔都进行了测试，同时保证alpha不能等于0或者C
            '''
            if (labelMat[i]*Ei < -toler and alphas[i] < C) or (labelMat[i]*Ei > toler and alphas[i] > 0):
                j = selectJrand(i, m)  # 选取第二个alpha值
                fXj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b  # 得到该样本的预测类别
                Ej = fXj - float(labelMat[j])  # 计算该样本的误差Ej

                # 开辟新空间保存alphas原有的i，j两个值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 计算L和H阈值，用于保证alpha在0到C之间，alpha值都>=0
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue

                # 计算得到alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta  # 根据eta对alpha[j]进行调整
                alphas[j] = clipAlpha(alphas[j], H, L)

                # 检验alpha[j]的改变，如果改变很小，则退出循环
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 对alpha[i]进行同样改变，只是改变方向相反（为了满足【全相加=0】约束）
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])

                # 给两个alpha【i和j】设置一个常数项
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2.0

                # 如果前面的continue语句均未执行，则说明成功修改了一对alpha值
                alphaPairsChanged += 1
                print("【iter: %d i:%d, pairs changed %d】" % (iter, i, alphaPairsChanged))

        # 外循环内，检测本次内循环是否更新了alpha，从而对iter次数进行更改
        if alphaPairsChanged == 0:
            iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


# 用于清理简化SMO算法的数据结构（完整版算法中用到，增加了缓存误差的矩阵）
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # 缓存误差（第一列是有效性标志，第二列是实际误差）

'''
    对于给定的alpha，计算误差值E
    oS : 相应类的对象
    k  : 样本下标值
'''
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.X * oS.X[k, :].T) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 选择第二个alpha（内循环中的alpha值）
def selectJ(i, oS, Ei):  # this is the second choice -heurstic（启发式策略）, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 标志位置1，选择可以带来最误差变化步长的alpha值
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   # loop through valid Ecache values and find the one that maximizes delta E
            if k == i:
                continue  # 跳过i的循环
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:  # 选择最大步长
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:   # in this case (first time around【第一轮没有找到有效的误差值】) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 计算误差值，并存入缓存之中
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 完整版SMO算法的 优化例程
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # 带有启发式策略
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)  #
        updateEk(oS, j)  # 更新误差缓存
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])  # i和j更新的量一样
        updateEk(oS, i)  # 更新误差缓存，这里的更新针对i的相反方向
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
            oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
            oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

'''
    完整版本的 Platt SMO 算法
'''
def smoP(dataMatIn, classLabels, C, toler, maxIter):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)  # 定义数据结构
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    while (iter < maxIter) and ( (alphaPairsChanged > 0) or (entireSet) ):
        alphaPairsChanged = 0
        if entireSet:   # 遍历所有的值
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历非边界的值
            nonBoundIs = nonzero( (oS.alphas.A > 0) * (oS.alphas.A < C) )[0]  # 可以找到不在边界上的样本的索引序号
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


# 根据alpha值计算参数w
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i, :].T)
    return w


# ===========================================================================================
# =============================== 在线性基础上，使用核函数 ====================================
# ===========================================================================================

class optStructK:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag
        self.K = mat(zeros((self.m, self.m)))  # 存放核函数内容
        for i in range(self.m):  # 使用核函数
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


'''
    calc the kernel or transform data to a higher dimensional space
    计算核函数，把数据转移到高维空间
'''
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':  # 线性核函数
        K = X * A.T
    elif kTup[0] == 'rbf':  # 径向基核函数
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp( K / (-1*kTup[1]**2) )  # 【元素对】之间的除法
    else:
        raise NameError('Houston We Have a Problem That Kernel is not recognized')
    return K


# 测试核函数（高斯径向基）
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('./testSetRBF.txt')

    # 设置C=200，同时使用高斯径向基函数
    b, alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  # 训练得到参数

    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]  # 获取相应的支持向量索引
    sVs = datMat[svInd]  # 得到所有的支持向量
    labelSV = labelMat[svInd]  # 获取支持向量对应的标签
    print("/nthere are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)

    # 测试1
    errorCount = 0
    for i in range(m):  # 开始使用核函数进行分类
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')

    # 测试2
    errorCount = 0
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))  # 得到核函数转换后的数据
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b  # 利用支持向量进行分类
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))


def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache, the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

'''
    full Platt SMO 加入核函数
'''
def smoPK(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStructK(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)

    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerLK(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLK(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


'''
    手写识别问题
    图片转化为向量
'''
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# 加载图像数据
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

'''
    这里的SVM算法只做二分类处理，knn可以直接做多分类
'''
def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('./trainingDigits')
    b, alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


if __name__ == '__main__':

    # 1、加载数据，测试辅助函数
    dataArr, labelArr = loadDataSet('./testSet.txt')  # label的标签是-1和+1，而不是传统的0和1
    print(len(labelArr), len(dataArr), len(dataArr[0]))

    # 2、简化版本的SMO算法
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # print(b, len(alphas), shape(alphas[alphas > 0]))
    # for i in range(100):
    #     if alphas[i] != 0:
    #         print(dataArr[i], labelArr[i])  # 输出了4个支持向量
    # print(alphas)  # 基本上都为0，不为0的就是支持向量
    print('\n**********************************************************************\n')

    # 3、完整版本 Platt SMO 算法
    # b1, alphas1 = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # ws = calcWs(alphas1, dataArr, labelArr)
    # print(b1, len(alphas1), shape(alphas1[alphas1 > 0]))

    # 4、测试高斯径向基函数
    # testRbf()


    # 5、使用SVM处理手写识别图片
    testDigits()