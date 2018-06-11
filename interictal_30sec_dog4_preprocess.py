import numpy as np
import scipy.io
import cPickle as pickle

#new object to be made and saved in pickle file from input mat file
class NeuralData:
    def __init__(self, dataType, dog, trial, sequenceNum, samplingRate, dataArray):
        self.dataType = dataType
        self.dog = dog
        self.trial = trial
        self.sequenceNum = sequenceNum
        self.samplingRate = samplingRate
        self.dataArray = dataArray

#number of preictal examples for each dog - match with interictal for similar data amounts
preictalNum = {
    'Dog1' : 24,
    'Dog2' : 42,
    'Dog3' : 72,
    'Dog4' : 97,
    'Dog5' : 30
}

numDog = 4 #only using dog4
numSlicesOriginal = 6 #number of slices in each hour - 10 min slices
splitRatio = 20 #split each 10 min into how many pieces

totalNegativeExamples = 0

for i in [numDog]:
    trial = 0
    tempPreictalNum = preictalNum['Dog' + str(i)]
    for j in range(1, tempPreictalNum + 1):
        print j
        inFileName = 'Dog_' + str(i) + '\Dog_' + str(i) + '_interictal_segment_' + str(j).zfill(4) + '.mat'
        matData = scipy.io.loadmat(inFileName, struct_as_record = False, squeeze_me = True)
        tempStr = 'interictal_segment_' + str(j)
        if ((matData[tempStr].sequence == numSlicesOriginal) or (matData[tempStr].sequence == numSlicesOriginal - 1)): #last 10 minutes in recording
            dog = i
            samplingRate = matData[tempStr].sampling_frequency
            splitSize = matData[tempStr].data.shape[1]/splitRatio
            for k in range(splitRatio):
                sequenceNum = (matData[tempStr].sequence-1)*splitRatio + k #after dividing into smaller slices, calculate new sequence number (position in hour long recording)
                startInd = k*splitSize
                endInd = (k+1)*splitSize
                dataArray = matData[tempStr].data[:, startInd:endInd]
                instance = NeuralData('interictal', dog, trial, sequenceNum, samplingRate, dataArray)
                outFileName = 'TrainingData_10sec_dog4\interictal_' + str(totalNegativeExamples) + '.pkl'
                with open(outFileName, 'wb') as output:
                    pickle.dump(instance, output, pickle.HIGHEST_PROTOCOL)
                del instance
                totalNegativeExamples += 1
                trial += 1
