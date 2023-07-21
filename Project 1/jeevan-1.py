# Importing Libraries
import os
from collections import Counter
import math
import nltk
from nltk.tokenize import RegexpTokenizer as regexpTokenizer
from nltk.corpus import stopwords as stopWords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

# Main Loop
folderPath = "./P1/"
fileName = ["US_Inaugural_Addresses/" +
            LoopObjX for LoopObjX in list(os.listdir(folderPath+'/US_Inaugural_Addresses'))]
fileStemmedData = []
allWordsCounter = Counter()
fileWords = {}
stopWord = stopWords.words('english')
Tokenizer_Object = regexpTokenizer(r'[a-zA-Z]+')
stemObj = PorterStemmer()
for instantFile in list(os.listdir(folderPath+'/US_Inaugural_Addresses')):
    # read, tokeize , stop word removal and stemming file data
    fileStemT = [stemObj.stem(loopWord) for loopWord in [loopWord for loopWord in Tokenizer_Object.tokenize(open(
        folderPath+'/US_Inaugural_Addresses/'+instantFile, "r").read().lower()) if loopWord not in stopWord]]
    fileStemmedData.append(fileStemT)
    # all words
    loopCounter = Counter(fileStemT)
    fileWords["US_Inaugural_Addresses/"+instantFile] = loopCounter
    allWordsCounter = allWordsCounter + loopCounter

# IDF Function


def getidf(funcString):
    if allWordsCounter[funcString] == 0:
        return -1
    else:
        return math.log10(len(fileName) / [1 for loopFileD in fileStemmedData if funcString in loopFileD].count(1))


# finding tf weight
termFreWeight = {fileNameL: Counter() for fileNameL in fileName}
calWeight = {fileNameL: Counter() for fileNameL in fileName}
for fileNameL in fileName:
    for loopWord in fileWords[fileNameL]:
        termFreWeight[fileNameL][loopWord] = 1 + \
            math.log10(fileWords[fileNameL][loopWord])
        calWeight[fileNameL][loopWord] = termFreWeight[fileNameL][loopWord] * \
            getidf(loopWord)

# normalization of weights
tempVariableOverallWeight = Counter()
tempVariableOverallWeightDocument = {}
for fileNameL in fileName:
    tempVariableOverallWeight[fileNameL] = math.sqrt(sum(
        [(termFreWeight[fileNameL][loopWord]) ** 2 for loopWord in fileWords[fileNameL]]))
    tempVariableOverallWeightDocument[fileNameL] = math.sqrt(
        sum([(calWeight[fileNameL][loopWord]) ** 2 for loopWord in fileWords[fileNameL]]))
    for loopWord in fileWords[fileNameL]:
        termFreWeight[fileNameL][loopWord] = termFreWeight[fileNameL][loopWord] / \
            tempVariableOverallWeight[fileNameL]
        calWeight[fileNameL][loopWord] = calWeight[fileNameL][loopWord] / \
            tempVariableOverallWeightDocument[fileNameL]

# get weight function


def getweight(funcFileN, funcWordN):
    if funcWordN in calWeight["US_Inaugural_Addresses/" + funcFileN]:
        return calWeight["US_Inaugural_Addresses/" + funcFileN][funcWordN]
    else:
        return 0

# query function


def query(funcStringT):
    listCosineSimi = []
    finalFileIndex = None
    for fileNameL, fileDataL in zip(fileName, fileStemmedData):
        finalWordCollectedL = list(set(
            [stemObj.stem(loopWord) for loopWord in funcStringT.lower().split()] + fileDataL))
        termFreqWei = [1 if interLoopObj in [stemObj.stem(loopWord) for loopWord in funcStringT.lower(
        ).split()] else 0 for interLoopObj in finalWordCollectedL]
        queryIdf = [0 if interLoopObj == -1 else interLoopObj for interLoopObj in [
            getidf(outLoopObj) for outLoopObj in finalWordCollectedL]]
        queryFinalWei = [termFreqWei[LoopObjX] * queryIdf[LoopObjX]
                         for LoopObjX in range(len(finalWordCollectedL))]
        Document_Normalized_Weight = [
            0 if interLoopObj not in fileDataL else termFreWeight[fileNameL][interLoopObj] for interLoopObj in finalWordCollectedL]
        listCosineSimi.append(sum([queryFinalWei[LoopObjX] * Document_Normalized_Weight[LoopObjX]
                              for LoopObjX in range(len(finalWordCollectedL))]))
        finalFileIndex = listCosineSimi.index(max(listCosineSimi))
    if listCosineSimi[finalFileIndex] == 0.0:
        return "Need More Fetching of Words", 0.0
    else:
        return fileName[finalFileIndex].split("/")[1], listCosineSimi[finalFileIndex]


# Questions
print("%.12f" % getidf('british'))  # 0.698970004336
print("%.12f" % getidf('union'))  # 0.062147906749
print("%.12f" % getidf('war'))  # 0.096910013008
print("%.12f" % getidf('power'))  # 0.029963223377
print("%.12f" % getidf('great'))  # 0.096910013008
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt', 'arrive'))  # 0.000000000000
print("%.12f" % getweight('07_madison_1813.txt', 'war'))  # 0.016131105316
print("%.12f" % getweight('12_jackson_1833.txt', 'union'))  # 0.011635243734
print("%.12f" % getweight('09_monroe_1821.txt', 'great'))  # 0.011052801776
print("%.12f" % getweight('05_jefferson_1805.txt', 'public'))  # 0.004181740452
print("--------------")
# (03_adams_john_1797.txt, 0.024425456076)
print("(%s, %.12f)" % query("pleasing people"))
# (07_madison_1813.txt, 0.063453716362)
print("(%s, %.12f)" % query("british war"))
# (05_jefferson_1805.txt, 0.054284470366)
print("(%s, %.12f)" % query("false public"))
# (13_van_buren_1837.txt, 0.007567974746)
print("(%s, %.12f)" % query("people institutions"))
# (02_washington_1793.txt, 0.202767835033)
print("(%s, %.12f)" % query("violated willingly"))
