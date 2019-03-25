'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# 创建候选集合C1
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()  # 排序
    return list(map(frozenset, C1))  # 对C1中每个项，使用frozenset（不变集合）进行包装，因为需要作为字典的key使用

'''
    从候选集C1生成L1
    返回的supportData，包含支持度的值，用于后续使用
'''
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:  # 遍历数据集中交易记录
        for can in Ck:  # 遍历候选集
            if can.issubset(tid):
                if can not in ssCnt.keys():  # 计算出现次数
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = float(len(D))  # 数据集中交易记录数
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems  # 计算该项集的支持度
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


'''
    创建下一级别的候选集合Ck
    Lk ： 频繁项集列表Lk
    k  ： 项集元素个数k
'''
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]  # 前k-2个项相同时，将两个集合合并
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # 如果前k-2个项相同
                retList.append(Lk[i] | Lk[j])  # 取并集
    return retList


# apriori主函数
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)  # 生成最初的单项候选集合
    D = list(map(set, dataSet))  # 数据集
    L1, supportData = scanD(D, C1, minSupport)  # 得到L1频繁项集
    L = [L1]  # L装所有的频繁项结果
    k = 2  # 算法中的参数，利用k-2，可以省去很多重复计算
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)  # 由L[k-2]生成Ck（其实L[k-2]是Lk-1，放到列表中差了一项）
        Lk, supK = scanD(D, Ck, minSupport)  # 扫描数据集，从Ck得到Lk（候选集得到频繁项集）
        supportData.update(supK)  # 字典加入新内容
        L.append(Lk)  # 追加频繁项集
        k += 1
    return L, supportData


'''
    关联规则生成函数（还利用apriori原则）
    L           :  所有的频繁项集
    supportData ： 每个频繁项对应的支持度（supportData is a dict coming from scanD）
    minConf     ： 最小置信度
'''
def generateRules(L, supportData, minConf=0.7):  #
    bigRuleList = []  # 存放规则结果
    for i in range(1, len(L)):  # 注意从1开始，只获取有两个或者更多元素的频繁项
        for freqSet in L[i]:  # 遍历每一个项
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

'''
    找到满足最小置信度的规则
    H : 存放频繁项集freqSet中的每一项
'''
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # 存放结果规则
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]  # 计算置信度（注意差集用法）
        if conf >= minConf: 
            print(freqSet-conseq, '-->', conseq, '|conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

'''
    从多于2个元素的项集中，得到更多的规则（注意组合结果）
    H : 可以出现在规则右侧的元素列表
'''
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])  # 频繁集大小
    calcConf(freqSet, H, supportData, brl, minConf)
    if len(freqSet) > (m + 1):  # 尝试进一步合并
        Hmp1 = aprioriGen(H, m+1)  # 创建H（m+1）候选集合，作为右侧（后件，后件在不断增大）
        if len(Hmp1) > 1:  # 如果多于两个集合，则可以考虑进一步合并
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)




# =================================================================================
# 国会投票的代码【无效】，没有对应api

from time import sleep
# from votesmart import votesmart
# votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
# votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)  # api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning

def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f\n" % ruleTup[2])



if __name__ == '__main__':

    # 1、加载数据，使用算法
    data_set = loadDataSet()
    print(data_set)
    C1 = createC1(data_set)
    print('候选集合c1', C1)
    D = list(map(set, data_set))
    print('数据集合', D)
    L1, suppData0 = scanD(D, C1, 0.5)
    print(L1)

    # 2、完整的apriori算法
    L, suppData = apriori(data_set, minSupport=0.2)
    print('\n', L)  # 最后会带有一个空列表，代表最后一个空组合

    # 3、关联规则生成（这里改动了下源码，源码中有些关联规则缺失）
    rules = generateRules(L, suppData, minConf=0.7)
    print('\n', rules)

    # 4、算法实例 --- 发现毒蘑菇的相似特征
    mush_data_set = [line.split() for line in open('mushroom.dat').readlines()]
    L, suppData = apriori(mush_data_set, minSupport=0.3)
    for item in L[1]:
        if item.intersection('2'):
            print(item)
    for item in L[2]:
        if item.intersection('2'):
            print(item)
