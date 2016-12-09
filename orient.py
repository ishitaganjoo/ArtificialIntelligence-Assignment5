from collections import defaultdict
import sys
from operator import sub
from random import randint
import math

def dd():
    return defaultdict(int)

class orient:

    def __init__(self):
        self.trainOrientationD = defaultdict(int)
        self.trainPhotoIdD = defaultdict(int)
        self.pixelVecLL = []
        self.confusionMatrix = defaultdict(dd)
        self.decStumpScore = defaultdict(int)
        self.weightList = []
        self.indicesList = []
        self.stumpAccuracyMatrix = []
        self.stumpCoordinateMatrix = []

    def readTrain(self, trainFile):
        fp = open(trainFile, 'r')
        idx = 0
        for eachImage in fp:
            data = [word for word in eachImage.split()]
            #print data[2::]
            res = map(int, data[2::])
            self.pixelVecLL += [res]
            self.trainPhotoIdD[idx] = data[0]
            self.trainOrientationD[idx] = int(data[1])
            idx += 1

        '''
        print self.trainPhotoIdD
        print ""
        print self.trainOrientationD
        print ""
        print self.pixelVecLL
        #print lines
        '''
    def nearest(self, testFile):
        fp = open(testFile, 'r')
        #mainIdx = 0
        totalCount = 0
        correctCount = 0
        for eachTestImage in fp:
            data = [word for word in eachTestImage.split()]
            imageName = data[0]
            imageOrient = int(data[1])
            #print mainIdx
            #mainIdx += 1
            idx = 0
            minEul = float('inf')
            res = map(int, data[2::])
            for eachTrainImage in self.pixelVecLL:
                #currEul = self.calcEuclidean(eachTrainImage,res)
                total = 0
                for eachTrain, eachTest in zip(eachTrainImage, res):
                    total += ((eachTrain - eachTest) ** 2)

                currEul = total
                if currEul < minEul:
                    minEul = currEul
                    maxIdx = idx
                idx += 1

            self.confusionMatrix[self.trainOrientationD[maxIdx]][imageOrient] += 1

            if self.trainOrientationD[maxIdx] == imageOrient:
                correctCount += 1
            totalCount += 1

        print "Accuracy is:", (correctCount/float(totalCount)) * 100

    def calcEuclidean(self, trainData, testData):
        #print testData
        subList = map(sub,trainData,testData)
        sqrList = map(lambda x: x ** 2, subList)
        sumList = sum(sqrList)
        #print subList
        return sumList ** 0.5

    def printConfusionMatrix(self):
        print " ".ljust(15) + "0".ljust(15) + "90".ljust(15) + "180".ljust(15) + "270".ljust(15)
        print "0".ljust(15) + str(self.confusionMatrix[0][0]).ljust(15) \
                            + str(self.confusionMatrix[0][90]).ljust(15) \
                            + str(self.confusionMatrix[0][180]).ljust(15) \
                            + str(self.confusionMatrix[0][270]).ljust(15)
        print "90".ljust(15) + str(self.confusionMatrix[90][0]).ljust(15) \
                             + str(self.confusionMatrix[90][90]).ljust(15) \
                             + str(self.confusionMatrix[90][180]).ljust(15) \
                             + str(self.confusionMatrix[90][270]).ljust(15)
        print "180".ljust(15) + str(self.confusionMatrix[180][0]).ljust(15) \
                              + str(self.confusionMatrix[180][90]).ljust(15) \
                              + str(self.confusionMatrix[180][180]).ljust(15) \
                              + str(self.confusionMatrix[180][270]).ljust(15)
        print "270".ljust(15) + str(self.confusionMatrix[270][0]).ljust(15) \
                              + str(self.confusionMatrix[270][90]).ljust(15) \
                              + str(self.confusionMatrix[270][180]).ljust(15) \
                              + str(self.confusionMatrix[270][270]).ljust(15)

    def adaboost(self,rotInd,stumps):
        print "In adaboost"
        rot = [0,90,180,270]
        self.decStumpScore = defaultdict(int)
        value, beta, maxVal,maxX,maxY = 0,0,0,0,0
        n = len(self.pixelVecLL)
        tempIndices=[]
        for s in range(0, stumps):
            print "stumps", s
            #do this after first stump
            if s==0:
                self.weightList = [1/float(n)]*n
                #print "n is ", n
                #print "weightList",self.weightList
            else:
                for l in range(0,len(self.weightList)):
                    if l not in self.indicesList:
                        #update weight
                        self.weightList[l]*=beta
                sumWeights = sum(self.weightList)
                #print("sum of list",sumWeights)
                self.weightList[:]= [x/float(sumWeights) for x in self.weightList]
                self.indicesList = []
                #print "weightList",self.weightList
            for i in range(0,1000):
                totalCnt = 0
                countRot = 0
                countNonRot = 0
                x = randint(0,191)
                y = randint(0,191)
                if x!=y and (x,y) not in self.decStumpScore:
                    tempIndices = []
                    for j in range(0,n):
                        if(self.pixelVecLL[j][x]>self.pixelVecLL[j][y]):
                            if(self.trainOrientationD[j]==rot[rotInd]):
                                #print j,
                                countRot+=self.weightList[j]
                                tempIndices.append(j)
                                totalCnt += 1
                        else:
                            if(self.trainOrientationD[j]!=rot[rotInd]):
                                #print j,
                                countNonRot+=self.weightList[j]
                                tempIndices.append(j)
                                totalCnt += 1

                    if(maxVal<countRot+countNonRot):
                        maxVal =countRot+countNonRot
                        maxX = x
                        maxY = y 
                        self.indicesList = tempIndices[:]
                        maxTotalCnt = totalCnt
            #print maxVal

            self.decStumpScore[(maxX,maxY)] = maxVal
                #print("countRot,countNonRot",countRot,countNonRot)
            #key,value = max(self.decStumpScore.iteritems(), key=lambda x:x[1])
            #print("maxValue and x,y is",maxVal, maxX, maxY)
            value = 1-maxVal
            #print "score dict",self.decStumpScore
            #print "count",maxTotalCnt
            beta = (value)/float(1-value)
            #print "beta", beta
            self.stumpAccuracyMatrix[rotInd][s] = math.log(1/beta)
            self.stumpCoordinateMatrix[rotInd][s] = (maxX,maxY)
                            
    def buildRotationMatrix(self,stumps):
        self.stumpAccuracyMatrix = [[0]*stumps for i in range(4)]
        self.stumpCoordinateMatrix = [[0]*stumps for i in range(4)]
        rot = [0,90,180,270]
        for r in range(0,len(rot)):
            self.adaboost(r,stumps)
                                   
        print "AccuracyMatrix is", self.stumpAccuracyMatrix
        print "CoordinateMatrix is", self.stumpCoordinateMatrix            
                        
    def testAdaboost(self,testFile,stumps):
        print "In testAdaboost"
        rot = [0,90,180,270]
        correctCount = 0
        totalCount = 0
        fp = open(testFile, 'r')
        for eachTestImage in fp:
            print totalCount
            data = [word for word in eachTestImage.split()]
            imageName = data[0]
            imageOrient = int(data[1])
            minEul = float('inf')
            res = map(int, data[2::])
            voteList = []
            for i in range(0,4):
                v = 0
                for j in range(0,stumps):
                    alpha = self.stumpAccuracyMatrix[i][j]
                    coordinate = self.stumpCoordinateMatrix[i][j]
                    if res[coordinate[0]]>res[coordinate[1]]:
                        v += alpha*1
                    else:
                        v += alpha*(-1)
                voteList.append(v)
            maxVote = max(voteList)
            #print maxVote
            #print voteList
            rotation = rot[voteList.index(maxVote)]            

            self.confusionMatrix[imageOrient][rotation] += 1

            if rotation == imageOrient:
                correctCount += 1
            totalCount += 1

        print "Accuracy is:", (correctCount/float(totalCount)) * 100
       
orientObj = orient()

if len(sys.argv) < 4:
    print "Please enter valid parameters..Exiting!!"
    sys.exit()

trainFile, testFile, method = sys.argv[1:4]

orientObj.readTrain(trainFile)
#orientObj.nearest(testFile)
#orientObj.printConfusionMatrix()
stumps = 3
orientObj.buildRotationMatrix(stumps)
orientObj.testAdaboost(testFile,stumps)
orientObj.printConfusionMatrix()
