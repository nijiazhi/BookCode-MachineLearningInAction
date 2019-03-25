'''
Created on Jun 14, 2011
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs: 
1. FP-tree (class treeNode)
2. header table (use dict)

This finds frequent itemsets similar to apriori but does not 
find association rules.  

@author: Peter
'''

'''
    FP树的类定义
    包含：节点名称、当前节点内容重复次数、链表指针、父节点指针、子节点集合
'''
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode      #needs to be updated
        self.children = {} 
    
    def inc(self, numOccur):  # 重复次数自增函数
        self.count += numOccur
        
    def disp(self, ind=1):  # 文本形式展示树（缩进代表树的深度）
        print('  '*ind, self.name, ' ', self.count)  # 注意空格的重复次数
        for child in self.children.values():
            child.disp(ind+1)


'''
    FP树构建函数（构建过程中会遍历两次数据集，这只有两次遍历就是比Apriori牛逼的地方）
'''
def createTree(dataSet, minSup=1): #create FP-tree from dataset but don't mine
    headerTable = {}  # 头指针列表
    # 整个建树过程遍历两次数据集合
    for trans in dataSet:  # 第一次遍历数据集，记录每个项的出现频次
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]  # 将出现次数记录在头指针列表中
    headerTable_copy = headerTable.copy()
    for k in headerTable_copy.keys():  # 在头指针列表中去除不满足最小支持度的项
        if headerTable_copy[k] < minSup:  # 这里对数据中每项进行判断，去除不满足最小支持的元素
            del(headerTable[k])

    freqItemSet = set(headerTable.keys())  # 每个频繁项
    print('freqItemSet: ', freqItemSet)

    if len(freqItemSet) == 0:
        return None, None  # 如果没有频繁项满足最小支持度，则退出
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]  # 对头指针列表进行指针链接
    print('headerTable: ', headerTable)

    retTree = treeNode('root Note', 1, None)  # 开始真正的建树过程
    for tranSet, count in dataSet.items():  # 第二次遍历数据集（这次需要使用每个频繁项的频率数量）
        localD = {}
        for item in tranSet:  # 根据全局频率，对事务中的每项进行排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]  # 当前频繁项将要链接的长度（频繁项的频次）
        if len(localD) > 0:  # 如果这条树枝长度大于0，就按照这条树枝的频次大小排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # 使用排序后的频率项对树进行填充，
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable  # 返回 树结构 和 头指针列表


'''
    FP树的增长函数
    items       ： 当前待处理的一条枝干
    inTree      ： 当前FP树的状态
    headerTable ： 头指针列表
    count       ： 当前事务集出现次数
'''
def updateTree(items, inTree, headerTable, count):
    # 判断排序后的items第一项，是否在树的子节点中
    if items[0] in inTree.children:  # 如果存在此节点
        inTree.children[items[0]].inc(count)  # 自增当前频繁项在数据的经过次数
    else:  # 如果不存在此节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)  # 需要将该项生成树节点，并加入树的子节点列表中
        # 如果头指针表，第一个位置没有链接（为None），就需要更新指针链表
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]  # 第一次链接上
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])  # 向头链表的后续指针中，添加新的树节点
    # 对items中剩余的项进行递归调用
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


'''
    该函数 确保节点链接 将targetNode指向树中该元素项的最后一个实例
    注意：函数中尽量别用递归，怕超过迭代调用的次数限制
'''
def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

# 根据给定节点leafNode向上遍历这棵树（prefixPath是待返回的前缀路径集合）
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

# 从treeNode遍历链表，直到结尾
def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:  # 路径长度大于1，把本身节点省略
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink  # 在链表中的下一个节点，知道为None
    return condPats

'''
    利用条件模式基，建立条件FP树，从而递归查找频繁项集
    freqItemList ： 最终的频繁项集列表
    preFix       :  存放上次挖掘的树的前缀项（最开始为空，一级一级的增加）
'''
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):

    # 首先依据频次对头指针列表进行排序，从小到大
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][1].count)]

    for basePat in bigL:  # 遍历头链表，从最小的频次开始
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        print('finalFrequent Item: ', newFreqSet)  # newFreqSet被加入最终结果freqItemList
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print('condPattBases :', basePat, condPattBases)

        # 这里重复使用createTree函数，目的是根据条件模式基 构建 该项的条件FP树
        myCondTree, myHead = createTree(condPattBases, minSup)
        print('head from conditional tree: ', myHead)
        if myHead != None:  # 如果条件树中还有元素，继续挖掘
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)  # 递归对条件树进行挖掘


# FP树 使用的简单数据集合
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


# 对基本数据进行格式化处理（对应项置为1）
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


'''----------------------------------------------------------------------------------------------------
【FP树挖掘实例 - 需要安装twitter模块和爬数据，这里不演示了】
import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1, 15):
        print("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList
'''

if __name__ == '__main__':

    # 1、测试FP树节点
    root_node = treeNode('root', 9, None)
    root_node.children['eye'] = treeNode('eye', 13, None)
    root_node.disp()
    root_node.children['andy'] = treeNode('andy', 11, None)
    root_node.disp()

    # 2、构建FP树
    minSup = 3
    sim_data = loadSimpDat()
    init_set = createInitSet(sim_data)
    my_fp_tree, myHeaderTab = createTree(init_set, minSup)  # 返回fp树和头链表
    my_fp_tree.disp()

    # 3、抽取条件模式基（条件模式基是以所查元素项为结尾的路径集合，其中每一条路径都是针对当前元素节点的前缀路径）
    t1 = findPrefixPath('x', myHeaderTab['x'][1])
    print('\n\n', t1)
    t2 = findPrefixPath('z', myHeaderTab['z'][1])
    print(t2)
    t3 = findPrefixPath('r', myHeaderTab['r'][1])
    print(t3, '\n\n')

    # 4、查找频繁项集
    myFreqList = []  # 创建频繁项集列表
    mineTree(my_fp_tree, myHeaderTab, minSup, set([]), myFreqList)
    print('\n\n', myFreqList)
