'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

# 加载数据
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)

'''
    PCA降维主程序
    topNfeat ： 应用的特征数量（降维效果）
'''
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # 去除均值
    covMat = cov(meanRemoved, rowvar=0)  # 协方差矩阵（一个对政治）
    print(covMat, '\n')
    eigVals, eigVects = linalg.eig(mat(covMat))  # 特征值
    print(eigVals, '\n\n', eigVects, '\n')
    eigValInd = argsort(eigVals)  # 从小到大排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 注意这里step为负，是从右向左切片
    print('选择的特征值索引 ： ', eigValInd)
    redEigVects = eigVects[:, eigValInd]  # 获取最大的那些特征值
    print('redEigVects shape', shape(redEigVects), '\n')
    lowDDataMat = meanRemoved * redEigVects  # 将数据转移到新的空间（返回的低维矩阵）
    print(redEigVects, redEigVects.T, redEigVects.I, '\n')
    # 重构原始数据（几个特征向量相当于新空间的坐标系）
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 移动坐标轴后的矩阵
    return lowDDataMat, reconMat


# 缺失值处理函数（千万注意nonzero的使用）
def replaceNanWithMean(): 
    datMat = loadDataSet('./secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # 计算所有非NaN的平均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # 将所有NaN置为平均值
    return datMat


if __name__ == '__main__':

    # 1、加载数据，使用pca降维
    data_mat = loadDataSet('testSet.txt')
    lowD_mat, recon_mat = pca(data_mat, 1)  # recon_mat是PCA处理后的数据
    print(shape(lowD_mat), shape(recon_mat))

    # 2、画出对比效果
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(recon_mat[:, 0].flatten().A[0], recon_mat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

    # 3、处理nan值，再使用pca
    data_mat = replaceNanWithMean()
    meanVals = mean(data_mat, axis=0)
    meanRemoved = data_mat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(covMat)
    print(eigVals)  # 注意观察有多少特征值是没有意义的（一般包含90%的信息就行）
