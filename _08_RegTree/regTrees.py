'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

# 加载数据集合
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)  # 最后一列是预测的回归值
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 将多有的元素映射为float类型
        dataMat.append(fltLine)
    return dataMat


# 二切分操作（利用数组过滤）
def binSplitDataSet(dataSet, feature, value):
    # print(dataSet.shape, type(dataSet))
    # mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]  # 去除后面的[0]，返回矩阵时会影响选择的行数
    mat0_index = nonzero(dataSet[:, feature] > value)
    mat0 = dataSet[mat0_index[0], :]  # nonzero得到的一个元组
    mat1_index = nonzero(dataSet[:, feature] <= value)
    mat1 = dataSet[mat1_index[0], :]
    return mat0, mat1


# 得到该叶节点样本的平均值，作为该叶节点的回归值
def regLeaf(dataSet):
    print(dataSet.shape, dataSet[:, -1][0, 0], dataSet[:, -1][0], dataSet[:, -1][0][0])
    return mean(dataSet[:, -1])


# 目标变量的平方误差
def regErr(dataSet):
    targe_list = dataSet[:, -1]
    var_val = var(targe_list)
    error = var_val * shape(dataSet)[0]
    return error


'''
    核心函数：找到数据的最佳二元切分方式
    函数内有三种情况不会切分，直接创建节点
    ops=(tolS，tolN)，存放参数，用于控制函数的停止时机
    tolS容许的误差下降值；
    tolN是切分的最少样本数量；
'''
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]

    # 如果所有值相等则退出（还是用set进行单一值判断）
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 退出条件1
        # print(leafType(dataSet))
        return None, leafType(dataSet)
    m, n = shape(dataSet)

    # 选择最佳特征的方法：利用误差函数计算得到
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        tmp_list = list(dataSet[:, featIndex].T.tolist())
        for splitVal in set(tmp_list[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差减少不大【S-bestS < tolS】，则退出
    if (S - bestS) < tolS: 
        return None, leafType(dataSet)  # 退出条件2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)  # 切分数据

    # 如果切分得到的数据集合很小，则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 退出条件3
        return None, leafType(dataSet)
    # 返回最优的分裂特征，以及该特征的切分值
    return bestIndex, bestValue

'''
    输入的数据dataSet是NumPy Mat，方便使用数据过滤
'''
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 选择最好的分割位置
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # print('切分位置和切分值 : ', feat, ' | ', val)
    if feat == None:  # 如果分割到了相应条件，则返回相应值
        return val

    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)  # 开始切分

    # 递归调用建树函数
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

# 通过类型判断树
def isTree(obj):
    return (type(obj).__name__=='dict')

# 返回树的平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

'''
    后剪枝策略
    tree     ： 待剪枝的树
    testData ： 剪枝所需的测试数据
'''
def prune(tree, testData):
    # 没有测试数据，对树进行塌陷处理
    if shape(testData)[0] == 0:
        return getMean(tree)  # 返回树的平均值

    # 如果分支是子树，就去对他们进行剪枝
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 如果两个分支是叶子节点了，则判断二者是否可以合并（比较合并前后的误差）
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


'''
    模型树使用
    主要是将数据集格式化为X和Y两种形式
'''
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))  # 注意第一列为1，对应X0
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]  # 填充Y
    xTx = X.T*X  # 线性回归结果

    if linalg.det(xTx) == 0.0:  # 判断可逆
        raise NameError('This matrix is singular, cannot do inverse, try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)  # 参数
    return ws, X, Y


# 创建线性模型，返回系数（用于叶子节点的生成）
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

# 在给定数据集上计算误差，可以利用它来得到最优划分
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

# 回归树 - 叶子节点预测
def regTreeEval(model, inDat):
    return float(model)


'''
 模型树 - 叶子节点预测
 叶子节点存储的model是参数，预测时需要乘以输入
'''
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]  # 特征数
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)


# 对给定输入数据进行预测（输入可以为单个数据点 或 行向量）
def treeForeCast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:  # 大于情况，走左侧树
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 多次调用treeForeCast函数，得到很多实验结果
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)  # 样本数量
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':

    # 1、加载数据，建立回归树
    myDat = loadDataSet('ex0.txt')
    myMat = mat(myDat)
    retTree_1 = createTree(myMat)  # 递归建树
    print(retTree_1)

    # 2、测试切分函数
    test_m = mat(eye(4))
    m0, m1 = binSplitDataSet(test_m, 1, 0.5)
    print(m0, '\n\n', m1)

    # 3、利用参数调整 进行预剪枝
    retTree_2 = createTree(myMat, ops=(0, 1))  # 调整建树参数
    print(retTree_2)
    myDat2 = loadDataSet('ex2.txt')
    myMat2 = mat(myDat2)
    tree_1 = createTree(myMat2)  # 递归建树
    print(tree_1)
    tree_2 = createTree(myMat2, ops=(10000, 4))  # 调整建树参数
    print(tree_2)

    # 4、后剪枝策略，合并树节点
    myDat2_test = loadDataSet('ex2test.txt')
    myMat2_test = mat(myDat2_test)
    myMat2 = mat(myDat2)
    myTree_2 = createTree(myMat2)
    print('\n\n', myTree_2)
    prune(myTree_2, myMat2_test)  # 调用剪枝函数
    print(myTree_2)

    # 5、模型树（叶节点是一个线性模型，存放其参数）
    myDat2 = loadDataSet('exp2.txt')
    myMat2 = mat(myDat2)
    model_tree = createTree(myMat2, leafType=modelLeaf, errType=modelErr, ops=(1, 10))
    print(model_tree)

    # 6、回归树 vs 模型树 vs 线性回归
    train_mat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    test_mat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    bike_reg_tree = createTree(train_mat, leafType=regLeaf, errType=regErr, ops=(1, 20))
    y_hat_reg = createForeCast(bike_reg_tree, test_mat[:, 0], modelEval=regTreeEval)  # 回归树得到预测结果
    reg_coff = corrcoef(y_hat_reg, test_mat[:, 1], rowvar=0)[0, 1]  # 返回协方差矩阵，第0行第1列的值

    bike_model_tree = createTree(train_mat, leafType=modelLeaf, errType=modelErr, ops=(1, 20))
    y_hat_model = createForeCast(bike_model_tree, test_mat[:, 0], modelEval=modelTreeEval)  # 模型树得到预测结果
    model_coff = corrcoef(y_hat_model, test_mat[:, 1], rowvar=0)[0, 1]

    y_hat_linear = []
    ws, X, Y = linearSolve(train_mat)
    for i in range(shape(test_mat)[0]):
        y_hat_linear.append(test_mat[i, 0]*ws[1, 0] + ws[0, 0])
    linear_coff = corrcoef(y_hat_linear, test_mat[:, 1], rowvar=0)[0, 1]

    print('\n\n\t\t', reg_coff, ' vs ', model_coff, 'vs', linear_coff)





