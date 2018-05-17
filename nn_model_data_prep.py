import numpy as np
import cPickle as pickle

preictal_examples = 800
interictal_examples = 780
nChannels = 16
numSamples = 11988
train_frac = 0.9

class NeuralData:
    def __init__(self, dataType, dog, trial, sequenceNum, samplingRate, dataArray):
        self.dataType = dataType
        self.dog = dog
        self.trial = trial
        self.sequenceNum = sequenceNum
        self.samplingRate = samplingRate
        self.dataArray = dataArray

allX = np.zeros((preictal_examples+interictal_examples, nChannels, numSamples))
allY = np.zeros((preictal_examples+interictal_examples, 1))
allSeq = np.zeros((preictal_examples+interictal_examples, 1))
for i in range(preictal_examples):
    inFileName = 'TrainingData\preictal_' + str(i) + '.pkl'
    with open(inFileName, 'rb') as f:
        x = pickle.load(f)
    allX[i] = np.array(x.dataArray)
    allY[i] = 1
    allSeq[i] = np.array(x.sequenceNum)

for i in range(interictal_examples):
    inFileName = 'TrainingData\interictal_' + str(i) + '.pkl'
    with open(inFileName, 'rb') as f:
        x = pickle.load(f)
    allX[i] = np.array(x.dataArray)
    allY[i] = 0
    allSeq[i] = np.array(x.sequenceNum)

trainIndices = np.random.rand(preictal_examples + interictal_examples) < train_frac
testIndices = ~trainIndices
X_train = allX[trainIndices]
X_test = allX[testIndices]
Y_train = allY[trainIndices]
Y_test = allY[testIndices]
Seq_train = allSeq[trainIndices]
Seq_test = allSeq[testIndices]

np.save('Xtrain.npy', X_train)
np.save('Xtest.npy', X_test)
np.save('Ytrain.npy', Y_train)
np.save('Ytest.npy', Y_test)
np.save('Seqtrain.npy', Seq_train)
np.save('Seqtest.npy', Seq_test)
