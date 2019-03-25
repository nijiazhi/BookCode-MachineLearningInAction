'''
Created on Mar 8, 2011

@author: Peter
'''
from numpy import *
from numpy import linalg as la


# 数据集1
def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


# 数据集2
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 欧式距离（用范数计算）
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

# 皮尔逊相关稀疏（归一化0到1）
def pearsSim(inA,inB):
    if len(inA) < 3 :
        return 1.0
    return 0.5 + 0.5*corrcoef(inA, inB, rowvar=0)[0][1]

# 余弦相似度
def cosSim(inA, inB):
    num = float(inA.T*inB)  # 内积结果
    denom = la.norm(inA)*la.norm(inB)  # 余弦公式反推
    return 0.5+0.5*(num/denom)


'''
    standEst计算 在给定相似度计算方法的条件下，用户对物品的估计评分值
    user ： 用户编号
    item ： 物品编号
'''
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]  # 物品数量
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):  # 遍历每列，即每个物品
        userRating = dataMat[user, j]  # 该用户对该物品的打分
        if userRating == 0:  # 如果评分为0，无意义，跳过
            continue
        # 寻找在【物品item】和【物品j】之间，同时评分的用户索引，记为overLap
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:  # 利用同时评分的用户，【物品item】和【物品j】之间的相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))

        simTotal += similarity  # 累计物品相似性
        ratSimTotal += similarity * userRating  # 物品相似性*该用户打分（用户大致能给出的分数）
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal  # 总得分/总相似性  （有点加权的意思）


# 推荐引擎函数
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 首先，寻找该用户user 没有评分的物品
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)  # 得到该用户 对 该物品的估计评分
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]  # 推荐得分最高前N个

'''
    svdEst函数对给定用户 的 给定物品得到评分估计值（内部还有SVD分解）
'''
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]  # 物品数量
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)  # svd分解
    Sig4 = mat(eye(4)*Sigma[:4])  # 取Sigma前四个元素
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  # 原来数据到低维空间转换
    # 剩余部分 就是 基于物品的推荐
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


'''
    打印矩阵
    由于降维后的矩阵包含浮点数，这里要给定一个阈值来变为深浅（01）
'''
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end='')
            else: print(0, end='')
        print('')

'''
    图像压缩函数
    numSV  ： 奇异值分解，压缩大小
    thresh ： 打印图像阈值
'''
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)  # 图像矩阵
    print("****original matrix******")
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):  # 根据压缩量构建新的Sigma对角矩阵
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV]*SigRecon*VT[:numSV, :]  # 构造降维之后的图像
    print("\n\n****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)


if __name__ == '__main__':

    # 1、尝试svd分解
    m = mat([[1, 1], [7, 7]])
    U, Sigma, VT = linalg.svd(m)  # Sigma以行向量形式返回
    print(U, '\n\n', Sigma, '\n\n', VT)

    # 2、尝试特征值分解
    eigVals, eigVects = linalg.eig(m)
    print('\n\n==============================\n')
    print(eigVects, '\n\n', eigVals)

    # 3、进一步尝试SVD
    data = loadExData()
    U, Sigma, VT = linalg.svd(data)
    print(Sigma)
    sigma_3 = mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])
    data_1 = U[:, :3] * sigma_3 * VT[:3, :]
    print(data_1)

    # 4、电影推荐的相似度计算
    m1 = mat(loadExData())
    euc_dis1 = ecludSim(m1[:, 0], m1[:, 4])
    euc_dis2 = ecludSim(m1[:, 0], m1[:, 0])
    print('\n\neuc_dis : ', euc_dis1, euc_dis2)

    cos_dis1 = cosSim(m1[:, 0], m1[:, 4])
    cos_dis2 = cosSim(m1[:, 0], m1[:, 0])
    print('cos_dis : ', cos_dis1, cos_dis2)

    pears_dis1 = pearsSim(m1[:, 0], m1[:, 4])
    pears_dis2 = pearsSim(m1[:, 0], m1[:, 0])
    print('pears_dis : ', pears_dis1, pears_dis2)

    # 5、基于物品的相似度推荐引擎
    m = mat(loadExData())
    m[0, 1] = m[0, 0] = m[1, 0] = m[2, 0] = 4
    m[3, 3] = 2
    r1 = recommend(m, 2)  # 开始使用推荐算法
    r2 = recommend(m, 2, simMeas=ecludSim)
    r3 = recommend(m, 2, simMeas=pearsSim)
    print('\n\n\n', r1, '\n\n', r2, '\n\n', r3)

    # 6、利用SVD提高推荐效果
    m = mat(loadExData2())
    U, Sigma, VT = linalg.svd(m)
    print('\n\n', Sigma)  # 寻找达到总能量90%的索引位置
    sig2 = Sigma**2
    # 分析能量后，发现可以降到3维
    print(sum(sig2), " | ", sum(sig2)*0.9, " | ", sum(sig2[:2]), " | ", sum(sig2[:3]))
    # 降到3维后，进行低维度下的推荐
    r1 = recommend(m, 2, estMethod=svdEst)
    r2 = recommend(m, 2, estMethod=svdEst, simMeas=pearsSim)
    r3 = recommend(m, 2, estMethod=svdEst, simMeas=ecludSim)
    print('\n\n\n', r1, '\n\n', r2, '\n\n', r3)

    # 6.1、原来算法效果
    r1 = recommend(m, 2)  # 开始使用推荐算法
    r2 = recommend(m, 2, simMeas=ecludSim)
    r3 = recommend(m, 2, simMeas=pearsSim)
    print('\n\n\n', r1, '\n\n', r2, '\n\n', r3)

    '''
        7、基于SVD的图像压缩（尽可能少的空间描绘图像）
           发现只需要两个奇异值和两个矩阵U, VT，就可以重构出原图（省空间）
    '''
    imgCompress(2)
