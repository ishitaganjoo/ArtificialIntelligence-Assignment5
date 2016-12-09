from collections import defaultdict
import sys
from operator import sub

def dd():
    return defaultdict(int)

class orient:

    def __init__(self):
        self.trainOrientationD = defaultdict(int)
        self.trainPhotoIdD = defaultdict(int)
        self.pixelVecLL = []
        self.confusionMatrix = defaultdict(dd)

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
        for eachTestImage in fp:
            data = [word for word in eachTestImage.split()]
            imageName = data[0]
            imageOrient = int(data[1])
            idx = 0
            minEul = float('inf')
            for eachTrainImage in self.pixelVecLL:
                res = map(int, data[2::])
                currEul = self.calcEuclidean(eachTrainImage,res)
                idx += 1
                if currEul < minEul:
                    minEul = currEul
                    maxIdx = idx
            #print idx, self.trainOrientationD[idx], imageOrient
            self.confusionMatrix[self.trainOrientationD[idx]][imageOrient] += 1
            #print self.trainOrientationD[idx], imageOrient,self.confusionMatrix[self.trainOrientationD[idx]][imageOrient]

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


orientObj = orient()

if len(sys.argv) < 4:
    print "Please enter valid parameters..Exiting!!"
    sys.exit()

trainFile, testFile, method = sys.argv[1:4]

orientObj.readTrain(trainFile)
orientObj.nearest(testFile)
orientObj.printConfusionMatrix()
