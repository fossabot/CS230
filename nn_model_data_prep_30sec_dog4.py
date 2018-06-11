import numpy as np
import cPickle as pickle

#number of examples of each category for baseline model
preictal_examples = 680
interictal_examples = 640

#dimension of data
nChannels = 16
numSamples = 11988

#data splitting between train and test
train_frac = 0.9

#type of object in each pickle file
class NeuralData:
    def __init__(self, dataType, dog, trial, sequenceNum, samplingRate, dataArray):
        self.dataType = dataType
        self.dog = dog
        self.trial = trial
        self.sequenceNum = sequenceNum
        self.samplingRate = samplingRate
        self.dataArray = dataArray

num_examples = preictal_examples + interictal_examples # for dog 4

#combine all data points into 3 arrays
allX = np.zeros((num_examples, numSamples, nChannels))
allY = np.zeros((num_examples, 1))
allSeq = np.zeros((num_examples, 1))

j = 0;

#loop through pickle files for preictal examples
for i in range(preictal_examples):
    inFileName = 'TrainingData_10sec_dog4\preictal_' + str(i) + '.pkl'
    with open(inFileName, 'rb') as f:
        x = pickle.load(f)
    print i
    if (x.dog == 4):
        allX[j, :, :] = np.array(np.transpose(x.dataArray))
        allY[j] = 1
        allSeq[j] = np.array(x.sequenceNum)
        j = j + 1;

#loop through pickle files for interictal examples
for i in range(interictal_examples):
    inFileName = 'TrainingData_10sec_dog4\interictal_' + str(i) + '.pkl'
    with open(inFileName, 'rb') as f:
        x = pickle.load(f)
    print i
    if (x.dog == 4):
        allX[j, :, :] = np.array(np.transpose(x.dataArray))
        allY[j] = 0
        allSeq[j] = np.array(x.sequenceNum)
        j = j + 1

#split the data into train and test
trainIndices = np.random.rand(num_examples) < train_frac
testIndices = ~trainIndices
X_train = allX[trainIndices]
X_test = allX[testIndices]
Y_train = allY[trainIndices]
Y_test = allY[testIndices]
Seq_train = allSeq[trainIndices]
Seq_test = allSeq[testIndices]

#save to output files
np.save('Xtrain4.npy', X_train)
X_train_fft = np.fft.fft(X_train, axis = 1)
np.save('Xtrain4_fft.npy', abs(X_train_fft))
np.save('Xtest4.npy', X_test)
X_train_fft = np.fft.fft(X_test, axis = 1)
np.save('Xtest4_fft.npy', abs(X_test_fft))
np.save('Ytrain4.npy', Y_train)
np.save('Ytest4.npy', Y_test)
np.save('Seqtrain4.npy', Seq_train)
np.save('Seqtest4.npy', Seq_test)
