from collections import defaultdict
import sys
from operator import sub
from random import randint
from random import randrange
import math
import time

def dd():
    return defaultdict(int)

class orient:

    def __init__(self):
        self.trainOrientD = defaultdict(int)
        self.trainPhotoIdD = defaultdict(int)
        self.pixelVecLL = []
        self.confusionMatrix = defaultdict(dd)
        self.decStumpScore = defaultdict(int)
        self.weightList = []
        self.indicesList = []
        self.stumpAccuracyMatrix = []
        self.stumpCoordinateMatrix = []
        self.weightsD = defaultdict(float)

    def readTrain(self, trainFile):
        fp = open(trainFile, 'r')
        idx = 0
        for eachImage in fp:
            data = [word for word in eachImage.split()]
            #print data[2::]
            res = map(int, data[2::])
            self.pixelVecLL += [res]
            self.trainPhotoIdD[idx] = data[0]
            self.trainOrientD[idx] = int(data[1])
            idx += 1

    def nearest(self, testFile):
        fp = open(testFile, 'r')
        totalCount = 0
        correctCount = 0
        for eachTestImage in fp:
            data = [word for word in eachTestImage.split()]
            imageName = data[0]
            imageOrient = int(data[1])
            idx = 0
            minEul = float('inf')
            res = map(int, data[2::])
            #res = [int(data[idx]) for idx in range(2,len(data))]
            for eachTrainImage in self.pixelVecLL:
                total = 0
                for eachTrain, eachTest in zip(eachTrainImage, res):
                    total += ((eachTrain - eachTest) ** 2)
                #for idx in range(len(eachTrainImage)):
                    #total += ((eachTrainImage[idx] - res[idx]) ** 2)

                currEul = total
                if currEul < minEul:
                    minEul = currEul
                    maxIdx = idx
                idx += 1

            self.confusionMatrix[self.trainOrientD[maxIdx]][imageOrient] += 1

            if self.trainOrientD[maxIdx] == imageOrient:
                correctCount += 1
            totalCount += 1

        print "Accuracy is:", (correctCount/float(totalCount)) * 100

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
        #tempCnt = 0 
        print "In adaboost"
        rot = [0,90,180,270]
        #self.decStumpScore = defaultdict(int)
        value, beta, maxVal, maxX, maxY = 0,0,0,0,0
        n = len(self.pixelVecLL)
        tempIndices=[]
        for s in range(0, stumps):
            if s == 0:
                self.weightList = [1.0 / n ] * n
            else:
                for l in range(0,len(self.weightList)):
                    if l in self.indicesList:
                        #update weight
                        self.weightList[l] *= beta
                sumWeights = sum(self.weightList)
                self.weightList[:]= [x/float(sumWeights) for x in self.weightList]
                self.indicesList = []
            maxVal = float('-inf')
            for i in range(0,500):
                totalCnt = 0
                countRot = 0
                countNonRot = 0
                x = randint(0,191)
                y = randint(0,191)
                #if x!=y and (x,y) not in self.decStumpScore:
                if x != y:
                    tempIndices = []
                    for j in range(0,n):
                        if(self.pixelVecLL[j][x] > self.pixelVecLL[j][y]):
                            if(self.trainOrientD[j] == rot[rotInd]):
                                countRot += self.weightList[j]
                                tempIndices.append(j)
                                totalCnt += 1
                        else:
                            if(self.trainOrientD[j] != rot[rotInd]):
                                countNonRot += self.weightList[j]
                                tempIndices.append(j)
                                totalCnt += 1

                    if(maxVal < countRot + countNonRot):
                        maxVal = countRot + countNonRot
                        maxX = x
                        maxY = y 
                        self.indicesList = tempIndices[:]
                        maxTotalCnt = totalCnt

            #self.decStumpScore[(maxX,maxY)] = maxVal
            value = 1 - maxVal
            beta = (value)/(1.0-value)
            self.stumpAccuracyMatrix[rotInd][s] = math.log(1.0/beta)
            self.stumpCoordinateMatrix[rotInd][s] = (maxX,maxY)
                            
    def buildRotationMatrix(self,stumps):
        self.stumpAccuracyMatrix = [[0]*stumps for i in range(4)]
        self.stumpCoordinateMatrix = [[0]*stumps for i in range(4)]
        rot = [0,90,180,270]
        for r in range(0,len(rot)):
            self.adaboost(r,stumps)
                                   
        print "stumps", stumps
        print "Confidence matrix is", self.stumpAccuracyMatrix
        print "CoordinateMatrix is", self.stumpCoordinateMatrix            
                        
    def testAdaboost(self,testFile,stumps):
        #print "In testAdaboost"
        rot = [0,90,180,270]
        correctCount = 0
        totalCount = 0
        fp = open(testFile, 'r')
        for eachTestImage in fp:
            #print totalCount
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
                        v += alpha
                    #else:
                        #v += alpha*(-1)
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

    def originalOutput(self, imageOrient):
        if imageOrient == 0:
            return [1.0, 0, 0, 0]
        elif imageOrient == 90:
            return [0, 1.0, 0, 0]
        elif imageOrient == 180:
            return [0, 0, 1.0, 0]
        elif imageOrient == 270:
            return [0, 0, 0, 1.0]

    def summation(self, inp, no, start, end):
        total = 0
        for idx in range(start, end):
            total += self.weightsD[(idx, no)] * inp[idx]

        return total

    def summation2(self, inp, no, start, end):
        total = 0
        for idx in range(start, end):
            total += self.weightsD[(no, idx)] * inp[idx]

        return total

    def actFunc(self, value):
        return 1.0/(1.0 + math.exp(-value))

    def actFuncDash(self, value):
        return self.actFunc(value) * (1.0 - self.actFunc(value))

    def randomWeights(self, hiddenNodeCnt, inputNodeCnt, outputNodeCnt):
        for inputNode in range(inputNodeCnt):
            for hiddenNode in range(hiddenNodeCnt):
                self.weightsD[(inputNode, hiddenNode + inputNodeCnt)] = randrange(-1,1) * (0.1)

        for hiddenNode in range(hiddenNodeCnt):
            for outputNode in range(outputNodeCnt):
                self.weightsD[(hiddenNode + inputNodeCnt, outputNode + inputNodeCnt + hiddenNodeCnt)] = randrange(-1,1) * (0.1)

    def updateWeights(self, alpha, actD, delta, inputNodeCnt, hiddenNodeCnt, outputNodeCnt):

        for x in range(inputNodeCnt):
            for y in range(hiddenNodeCnt):
                self.weightsD[(x, y + inputNodeCnt)] += actD[x] * delta[y + inputNodeCnt] * alpha

        for x in range(hiddenNodeCnt):
            for y in range(outputNodeCnt):
                self.weightsD[(x + inputNodeCnt, y + inputNodeCnt + hiddenNodeCnt)] += \
                                actD[x + inputNodeCnt] * delta[y + inputNodeCnt + hiddenNodeCnt] * alpha


    def trainNNet(self, hiddenNodeCnt, alpha, iterationCnt, inputNodeCnt, outputNodeCnt):
        #take random weights initially
        self.randomWeights(hiddenNodeCnt, inputNodeCnt, outputNodeCnt)

        #Run the loop iterationCnt times or until it converges
        for idx in range(iterationCnt):

            imageIdx = -1
            totalCnt = 0
            accurateCnt = 0
            #Consider each image
            for eachTrainImage in self.pixelVecLL:
                totalCnt += 1
                imageIdx += 1

                origOutput = self.originalOutput(self.trainOrientD[imageIdx])
                #Activation Dict
                actD = {}
                #input values in float
                inpD = {}
                #Error dict
                delta = {}

                #Forward propagation
                for eachVal in range(inputNodeCnt):
                    #print eachVal,inputNodeCnt
                    actD[eachVal] = self.pixelVecLL[imageIdx][eachVal] / 255.0
                for eachNeuron in range(inputNodeCnt, inputNodeCnt + hiddenNodeCnt):
                    inpD[eachNeuron] = self.summation(actD, eachNeuron, 0, inputNodeCnt) + 1
                    actD[eachNeuron] = self.actFunc(inpD[eachNeuron])

                actualOutput = []

                for eachNeuron in range(inputNodeCnt + hiddenNodeCnt, inputNodeCnt + hiddenNodeCnt + outputNodeCnt):
                    inpD[eachNeuron] = self.summation(actD, eachNeuron, inputNodeCnt, inputNodeCnt + hiddenNodeCnt) + 1
                    actD[eachNeuron] = self.actFunc(inpD[eachNeuron])
                    actualOutput.append(actD[eachNeuron])

                computedIdx = actualOutput.index(max(actualOutput))
                actualIdx = origOutput.index(1.0)

                if computedIdx == actualIdx:
                    accurateCnt += 1
    
                #back Propagation
                for eachNeuron in range(inputNodeCnt + hiddenNodeCnt, inputNodeCnt + hiddenNodeCnt + outputNodeCnt):
                    delta[eachNeuron] = self.actFuncDash(inpD[eachNeuron]) * \
                                        (origOutput[eachNeuron - inputNodeCnt - hiddenNodeCnt] - actD[eachNeuron])

                for eachNeuron in range(inputNodeCnt, inputNodeCnt + hiddenNodeCnt):
                    delta[eachNeuron] = self.actFuncDash(inpD[eachNeuron])* \
                                        self.summation2(delta, eachNeuron, inputNodeCnt + hiddenNodeCnt,\
                                               inputNodeCnt + hiddenNodeCnt + outputNodeCnt)

                #Code for update
                self.updateWeights(alpha, actD, delta, inputNodeCnt, hiddenNodeCnt, outputNodeCnt)
                #print "iteration " + str(idx) +\
                    #" | Train sample " + str(totalCnt) + " done | Accuracy: " + "%.4f" % ((accurateCnt/float(totalCnt))*100.0) + " %"
        print "Total Accuracy in train = ", accurateCnt, totalCnt, (accurateCnt / float(totalCnt)) * 100

    def testNNet(self, testFile, hiddenNodeCnt, alpha, iterationCnt, inputNodeCnt, outputNodeCnt):
        rot = [0, 90, 180, 270]
        actD = {}
        fp = open(testFile, 'r')
        totalCnt = 0
        accurateCnt = 0
        for eachTestImage in fp:
            data = [word for word in eachTestImage.split()]
            imageName = data[0]
            imageOrient = int(data[1])
            #res = map(int, data[2::])

            inpD = {}
            origOutput = self.originalOutput(imageOrient)
    
            for eachVal in range(inputNodeCnt):
                actD[eachVal] = float(data[2 + eachVal]) / 255

            for eachNeuron in range(inputNodeCnt, inputNodeCnt + hiddenNodeCnt):
                inpD[eachNeuron] = self.summation(actD, eachNeuron, 0, inputNodeCnt) + 1
                actD[eachNeuron] = self.actFunc(inpD[eachNeuron])

            actualOutput = []
            for eachNeuron in range(inputNodeCnt + hiddenNodeCnt, inputNodeCnt + hiddenNodeCnt + outputNodeCnt):
                inpD[eachNeuron] = self.summation(actD, eachNeuron, inputNodeCnt, inputNodeCnt + hiddenNodeCnt) + 1
                actD[eachNeuron] = self.actFunc(inpD[eachNeuron])
                actualOutput.append(actD[eachNeuron])

            computedIdx = actualOutput.index(max(actualOutput))
            actualIdx = origOutput.index(1.0)
            totalCnt += 1
            self.confusionMatrix[rot[actualIdx]][rot[computedIdx]] += 1

            if computedIdx == actualIdx:
                accurateCnt += 1

            #print "Test " + str(totalCnt) + " done "

        print "Accuracy in test: " + "%.4f" % ((accurateCnt/float(totalCnt))*100.0) + "%"


    def nnet(self,testFile,hiddenNodeCnt):
        rot = [0, 90, 180, 270]
        alpha = 0.1
        iterationCnt = 3
        inputNodeCnt = len(self.pixelVecLL[0])
        outputNodeCnt = len(rot)

        stime = time.time()
        self.trainNNet(hiddenNodeCnt, alpha, iterationCnt, inputNodeCnt, outputNodeCnt)
        etime = time.time()
        print "Time for train nnet= ", etime - stime , (etime - stime)/60

        stime = time.time()
        self.testNNet(testFile, hiddenNodeCnt, alpha, iterationCnt, inputNodeCnt, outputNodeCnt)
        etime = time.time()
        print "Time for test nnet= ", etime - stime , (etime - stime)/60

#MAIN PROGRAM
       
orientObj = orient()

if len(sys.argv) < 4:
    print "Please enter valid parameters..Exiting!!"
    sys.exit()

trainFile, testFile, method = sys.argv[1:4]

orientObj.readTrain(trainFile)

if method == "nearest":
    stime = time.time()
    orientObj.nearest(testFile)
    etime = time.time()
    print "Time for nearest: ", etime - stime, (etime - stime)/60
    orientObj.printConfusionMatrix()

elif method == "adaboost":
    if len(sys.argv) < 5:
        print "Enter valid stump count..Exiting!!"
        sys.exit()

    stumps = int(sys.argv[4])
    #stime = time.time()
    orientObj.buildRotationMatrix(stumps)
    #etime = time.time()
    #s1time = time.time()
    orientObj.testAdaboost(testFile,stumps)
    #e1time = time.time()
    #print "orientObj.buildRotationMatrix",etime - stime
    #print "orientObj.testAdaboost", e1time - s1time
    orientObj.printConfusionMatrix()

elif method == 'nnet':
    if len(sys.argv) < 5:
        print "Enter valid hidden count..Exiting!!"
        sys.exit()

    hiddenNodeCnt = int(sys.argv[4])
    orientObj.nnet(testFile,hiddenNodeCnt)
    orientObj.printConfusionMatrix()
