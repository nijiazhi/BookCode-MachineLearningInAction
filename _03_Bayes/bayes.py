# coding:utf-8
'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *


# 创建实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1表示消极的，滥用的
    return postingList, classVec


# 创建词袋（单词不重复）
def createVocabList(dataSet):
    vocabSet = set([])  # 创建空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 两个集合的并集
    return list(vocabSet)


# 生成词向量（词集模型）
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  # 创建全0向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


'''
    一、利用贝叶斯公式，训练分类器
    [1] 根据类别计算P(c)
    [2] 根据假设计算P(w|c)，又有w向量内单词相互独立，可以转换为P(w1,w2,w3...|c) = p(w1|c)p(w2|c)p(w3|c)...

    二、问题
    [1] 算法在初始化p0Num和p0Denom的过程中，没有直接初始化为0；因为若p(w1|c)=0，则会导致相乘之后的文档概率为0，称为加1平滑
    [2] 下溢出问题，很多大小的数乘在一起，会导致没有结果或者下溢出。所以这里用了log函数来避免影响，而不会带来概率的损失
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 文档的数量N
    numWords = len(trainMatrix[0])  # 单词（词库）的数量，特征数
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 类别为1的概率，因为是2分类，所以也可知道类别为0的概率
    p0Num = ones(numWords)  # 创建全1向量，数量为特征数，求解p(w1|c)过程中的初始化过程，作为分子使用
    p1Num = ones(numWords)  # 分子是一个numpy数组，它的元素个数等于词汇表大小
    p0Denom = 2.0  # 初始化概率的分母，
    p1Denom = 2.0
    for i in range(numTrainDocs):  # for循环每一篇文档，一旦某个词语在文档出现，则该词对应个数加1
        if trainCategory[i] == 1:  # 类别为1
            p1Num += trainMatrix[i]  # （分子）向量的累加，词语在文档出现，则该词对应个数加1
            p1Denom += sum(trainMatrix[i])  # 分母的累加，分母是该类别中的总词数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


# 贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 每个单独词语的条件概率相乘，得到文档的条件概率（独立性假设）
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

# 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 测试贝叶斯分类器
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    # 开始测试过程
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


# 字符串解析函数（输入大写字符串）
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 注意这种写法

# 垃圾邮件分类算法
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1, 26):
        wordList = textParse(open('./email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('./email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 创建词汇表

    trainingSet = list(range(50))
    testSet=[]  # 创建测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  # 删除训练集合对应内容

    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:  # 使用训练集的数据训练分类器
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # 使用分类器（计算所得的概率）做出分类
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))
    # return vocabList,fullText


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       


def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V


def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


if __name__ == '__main__':

    # 1、加载bayes数据
    list_posts, list_classes = loadDataSet()
    print(len(list_posts))

    # 2、生成词向量
    myVocabList = createVocabList(list_posts)  # 词袋
    print(len(myVocabList), myVocabList)
    post_vector = setOfWords2Vec(myVocabList, list_posts[0])  # 利用词袋将一个文档进行向量化表示
    print(len(post_vector), post_vector)

    # 3、利用贝叶斯公式计算，训练分类器
    trainMat = []
    for postinDoc in list_posts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    '''
        1、p0V和p1V是两个条件概率向量，对应每个词在该类别条件下的概率
        2、pAb是类别为1的概率
    '''
    p0V, p1V, pAb = trainNB0(trainMat, list_classes)

    # 4、使用贝叶斯分类器
    testingNB()

    # 5、区分词袋模型和词集模型（是否记录单词的出现次数）
    bagOfWords2VecMN(myVocabList, list_posts[0])

    # 6、垃圾邮件分类算法实验
    spamTest()