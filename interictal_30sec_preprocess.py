import numpy as np
import scipy.io
import cPickle as pickle

class NeuralData:
    def __init__(self, dataType, dog, trial, sequenceNum, samplingRate, dataArray):
        self.dataType = dataType
        self.dog = dog
        self.trial = trial
        self.sequenceNum = sequenceNum
        self.samplingRate = samplingRate
        self.dataArray = dataArray

preictalNum = {
    'Dog1' : 24,
    'Dog2' : 42,
    'Dog3' : 72,
    'Dog4' : 97,
    'Dog5' : 30
}

numDogs = 4 #not using dog5 for now because it only has 15 electrodes
numSlicesOriginal = 6
splitRatio = 20

totalPositiveExamples = 0

for i in range(1, numDogs+1):
    trial = 0
    tempPreictalNum = preictalNum['Dog' + str(i)]
    for j in range(1, tempPreictalNum + 1):
        print j
        inFileName = 'Dog_' + str(i) + '\Dog_' + str(i) + '_interictal_segment_' + str(j).zfill(4) + '.mat'
        matData = scipy.io.loadmat(inFileName, struct_as_record = False, squeeze_me = True)
        tempStr = 'interictal_segment_' + str(j)
        if (matData[tempStr].sequence == numSlicesOriginal): #last 10 minutes
            dog = i
            samplingRate = matData[tempStr].sampling_frequency
            splitSize = matData[tempStr].data.shape[1]/splitRatio
            for k in range(splitRatio):
                sequenceNum = (matData[tempStr].sequence-1)*splitRatio + k
                startInd = k*splitSize
                endInd = (k+1)*splitSize
                #endInd = (None if (k == (splitRatio - 1)) else ((k+1)*splitSize))
                dataArray = matData[tempStr].data[:, startInd:endInd]
                instance = NeuralData('interictal', dog, trial, sequenceNum, samplingRate, dataArray)
                outFileName = 'TrainingData\interictal_' + str(totalPositiveExamples) + '.pkl'
                with open(outFileName, 'wb') as output:
                    pickle.dump(instance, output, pickle.HIGHEST_PROTOCOL)
                del instance
                totalPositiveExamples += 1
                trial += 1
