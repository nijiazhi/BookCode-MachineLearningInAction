'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *
from time import sleep
import matplotlib.pyplot as plt


# 加载数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1  # 获取特征数量
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 标准的回归过程
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  # 行列式为0，不可逆
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


# 局部加权线性回归（针对一个点）
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]  # m为样本数量
    weights = mat( eye(m) )  # 样本数量的对角矩阵
    for j in range(m):  # 按照样本点到针对待测点的距离，更新权重
        diffMat = testPoint - xMat[j, :]  # 计算距离
        weights[j, j] = exp(diffMat*diffMat.T/(-2.0*k**2))  # 使用高斯核更新权重
    xTx = xMat.T * (weights * xMat)  # 计算更新权重后的数据矩阵
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))  # 根据公式得到最终决策边界权重
    return testPoint * ws


'''
    遍历样本集合中所有点，对每个点应用局部加权回归
    k : 高斯核参数，越小分布越集中
'''
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)  # 初始化y的估计值
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# 获取【局部加权回归图像】的结果
# 注意需要对x进行排序，不然会乱
def lwlrTestPlot(xArr, yArr, k=1.0):
    yHat = zeros(shape(yArr))
    xCopy = mat(xArr)
    xCopy.sort(0) # 对第二维度进行排序？
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy


# 均方误差计算
def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()


'''
    岭回归
    lam : 高斯核的参数，越小分布越集中
'''
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)  # 求逆后计算
    return ws


# 测试岭回归
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat=mat(yArr).T
    yMean = mean(yMat, 0)  # 取均值，数据标准化
    yMat = yMat - yMean  # 所有数减去均值，得到的新列满足均值为0
    xMeans = mean(xMat, 0)  # 计算均值
    xVar = var(xMat, 0)  # 计算方差
    xMat = (xMat - xMeans)/xVar  # 标准化公式（减去均值，除以方差）

    numTestPts = 30  # 测试30组不同的岭回归参数
    wMat = zeros((numTestPts, shape(xMat)[1]))  # 初始化参数矩阵
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))  # 不断改变岭回归的参数值
        wMat[i, :] = ws.T  # 第i行结果
    return wMat


# 对输入的数据矩阵进行标准化（针对每列，即每个特征）
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)   # 计算均值（对每列）
    inVar = var(inMat, 0)      # 计算方差
    inMat = (inMat - inMeans)/inVar
    return inMat

'''
    逐步前向回归（贪心思想）
    numIt ： 迭代轮数
    eps   ： 步长，控制增减的大小
'''
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # 对y进行标准化，效果会好些
    xMat = regularize(xMat)  # 对X进行标准化

    m, n = shape(xMat)
    ws = zeros((n, 1))
    wsMax = ws.copy()

    for i in range(numIt):
        print("当前参数向量：", ws.T)
        lowestError = inf  # 初始化Error为正无穷
        for j in range(n):  # 针对每个特征
            for sign in [-1, 1]:  # 控制增加或者减少特征
                wsTest = ws.copy()
                wsTest[j] += eps*sign  # 对特征值进行增加或者减小
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)  # 计算误差
                if rssE < lowestError:  # 更新误差
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
    print('stageWise done!')


'''

'''
def scrapePage(inFile, outFile, yr, numPce, origPrc):
   from BeautifulSoup import BeautifulSoup
   fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
   soup = BeautifulSoup(fr.read())
   i=1
   currentRow = soup.findAll('table', r="%d" % i)
   while(len(currentRow)!=0):
       title = currentRow[0].findAll('a')[1].text
       lwrTitle = title.lower()
       if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
           newFlag = 1.0
       else:
           newFlag = 0.0
       soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
       if len(soldUnicde)==0:
           print("item #%d did not sell" % i)
       else:
           soldPrice = currentRow[0].findAll('td')[4]
           priceStr = soldPrice.text
           priceStr = priceStr.replace('$','') #strips out $
           priceStr = priceStr.replace(',','') #strips out ,
           if len(soldPrice)>1:
               priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
           print("%s\t%d\t%s" % (priceStr,newFlag,title))
           fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
       i += 1
       currentRow = soup.findAll('table', r="%d" % i)
   fw.close()


'''
    购物数据的获取函数（网页无法访问）
'''
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    import json
    import urllib
    sleep(1)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


# 函数内部多次调用searchForSet
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


# 交叉验证测试岭回归
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal, 30))  # 误差矩阵（30列:所有的岭回归参数）
    for i in range(numVal):
        trainX=[]; trainY=[]  # 创建训练集和测试集容器
        testX = []; testY = []
        random.shuffle(indexList)  # 洗牌打乱所有样本序号
        for j in range(m):  # 利用所有样本需要的90%，创建训练集合
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # 获取训练得到的权重
        for k in range(30):  # 遍历所有的岭回归参数，得到误差
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX-meanTrain)/varTrain  # 标准化数据
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # 得到y的估计值
            errorMat[i, k] = rssError(yEst.T.A, array(testY))  # 得到误差并切存储
            print("交叉验证，当前误差：", errorMat[i, k])
    meanErrors = mean(errorMat, 0)  # 计算不同岭回归参数的平均表现
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]  # 找到最小的误差权重

    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat, 0); varX = var(xMat, 0)
    # 开始对数据进行还原
    unReg = bestWeights/varX  # 权重除方差？代表什么？
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1*sum(multiply(meanX, unReg)) + mean(yMat))


if __name__ == '__main__':

    # 1、执行最简单的直线回归函数
    xArr, yArr = loadDataSet('./ex0.txt')
    ws = standRegres(xArr, yArr)
    print(ws)

    # 2、画出简单直线回归的图像
    xMat = mat(xArr)
    yMat = mat(yArr).T
    print(xMat.shape, yMat.shape)
    # yHat = xMat * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat[:, 0].flatten().A[0], s=3)
    xCopy = xMat.copy()
    xCopy.sort(0)  # 按照列进行排序
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)

    # 3、利用相关系统判断回归的好坏
    yHat = xMat * ws
    cor = corrcoef(yHat.T, yMat.T)  # 输入需要是行向量
    print(cor)

    # 4、进行局部加权回归测试
    xArr, yArr = loadDataSet('./ex0.txt')
    print('\n\n一个待测点：', xArr[0], yArr[0])
    new_point_1 = lwlr(xArr[0], xArr, yArr, k=1)  # 对待测试点进行局部加权回归
    print(new_point_1)
    new_point_2 = lwlr(xArr[0], xArr, yArr, k=0.01)  # 对待测试点进行局部加权回归
    print(new_point_2)
    yHat_1 = lwlrTest(xArr, xArr, yArr, k=0.03)  # 对所有样本点进行局部加权回归
    xMat = mat(xArr)
    sort_index = xMat[:, 1].argsort(0)  # 按照列排序
    x_sort = xMat[sort_index][:, 0, :]
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(x_sort[:, 1], yHat_1[sort_index])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).flatten().A[0], s=2, color='red')

    # 5、岭回归测试
    abX, abY = loadDataSet('./abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)  # 获取岭回归参数矩阵
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)

    '''
        6、前向逐步线性回归
        【1】当eps=0.01时，可以观察结果，发现步长有些大
        【2】当eps=0.001时，再次得到新的结果
    '''
    # stageWise(abX, abY, eps=0.01, numIt=200)
    # stageWise(abX, abY, eps=0.001, numIt=5000)  # 结果比较稳定

    # 7、前向逐步线性回归 vs 最小二乘回归（OLS）
    xMat = regularize(mat(abX))
    yMat = mat(abY).T
    yMat = yMat - mean(yMat, 0)
    weights = standRegres(xMat, yMat.T)
    print("最小二乘回归：", weights)

    # 8、乐高玩具数据 建模回归预测
    # lgX = []
    # lgY = []
    # setDataCollect(lgX, lgY)

    # 最终画图
    plt.show()
