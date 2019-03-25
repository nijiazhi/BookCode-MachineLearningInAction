'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''

from numpy import *

# 加载数据
def loadDataSet(fileName):
    dataMat = []  # 最后一列为标签列
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 所有元素map到float类型（注意需要添加float）
        dataMat.append(fltLine)
    return dataMat

# 欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 另一种表达：la.norm(vecA-vecB)

# 对给定数据集选择包含k个随机质心的集合（k是质心个数）
def randCent(dataSet, k):
    n = shape(dataSet)[1]  # 特征数
    centroids = mat(zeros((k, n)))  # 创建质心矩阵
    for j in range(n):  # 在数据范围内，随机选择质心
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  # 填充质心矩阵
    return centroids

# kmeans 聚类算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  # 样本数量
    # 创建矩阵 存储每个点的簇分配结果（1列为簇索引值；2列为误差，这里即欧式距离）
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True  # 聚类标记
    while clusterChanged:
        clusterChanged = False  # 先对聚类标记置否
        for i in range(m):  # 对每个数据点，寻找最近的质心
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 如果质心发生改变，修改标志位
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print('centroids:\t', centroids)
        for cent in range(k):  # 更新质心位置（注意nonzero的使用）
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获取该簇中所有点
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 该簇的平均值
    return centroids, clusterAssment


# 二分K均值聚类算法（可以收敛到全局最小值）
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # 创建矩阵 存储每个点的簇分配结果
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 创建初始质心（均值）
    centList = [centroid0]  # 创建初始的簇
    for j in range(m):  # 计算初始化误差
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2

    while len(centList) < k:  # 控制簇的个数
        lowestSSE = inf
        for i in range(len(centList)):  # 遍历每个质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  # 得到当前质心簇内的所有样本点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 尝试对该簇进行聚类（聚类数目为2）
            sseSplit = sum(splitClustAss[:, 1])  # 计算聚类后的新误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 计算剩余样本点（非当前质心簇内的所有样本点）的误差
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:  # 比较误差，判断是否需要进行划分聚类
                bestCentToSplit = i
                bestNewCents = centroidMat  # 新的质心结果
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 更新 新加入的聚类结果
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 新聚类结果【1】作为第len(centList)个聚类质心索引
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 新聚类结果【0】作为原来位置i上的聚类质心索引
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 使用新的质心结果【0】代替原来位置
        centList.append(bestNewCents[1, :].tolist()[0])  # 加入新的质心结果【1】
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # 更新簇的分配结果（索引）和误差
    return mat(centList), clusterAssment


# 聚类实例：对地图上的点进行聚类
import urllib
import json

def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print(yahooApi)
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print ("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print ("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):  # 余弦球面距离
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0


import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__  == '__main__':

    # 1、加载数据，随机找中心
    data_mat = mat(loadDataSet('testSet.txt'))
    min_v = min(data_mat[:, 0])
    max_v = max(data_mat[:, 0])
    center = randCent(data_mat, 2)
    dis = distEclud(data_mat[0], data_mat[1])
    print(min_v, max_v, '\n', center, '\ndis:', dis)

    # 2、k均值聚类算法
    myCentroids, clustAssing = kMeans(data_mat, 4)
    print(myCentroids, '\n##################\n')
    print(clustAssing)

    # 3、二分k均值聚类算法（可以收敛到全局最小值）
    data_mat_2 = mat(loadDataSet('testSet2.txt'))
    cenList, myNewAssments = biKmeans(data_mat_2, 3)
    print(cenList, '\n##################\n')
    print(myNewAssments)

    # 4、聚类实例
    geoResults = geoGrab('1 VA Center', 'August, ME')
    print('\n\n##################', geoResults)