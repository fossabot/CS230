import numpy as np
import math
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv2D, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

def load_dataset():
    X_train = np.load('Xtrain4_fft.npy')
    Y_train = np.load('Ytrain4.npy')
    X_test = np.load('Xtest4_fft.npy')
    Y_test = np.load('Ytest4.npy')
    return X_train, Y_train, X_test, Y_test

def model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape = input_shape)
    # Step 1: CONV layer
    X = Conv1D(16, 10, strides=10)(X_input)
    X = BatchNormalization()(X)

    # Step 2: First LSTM Layer
    X = LSTM(units = 48, return_sequences = True)(X)
    X = BatchNormalization()(X)

    # Step 3: Second LSTM Layer
    X = LSTM(units = 80, return_sequences = False)(X)
    X = Dropout(0.5)(X)

    # Step 4: Dense layer
    X = Dense(1, activation = "sigmoid")(X)
    model = Model(inputs = X_input, outputs = X)

    return model

X_train, Y_train, X_test, Y_test = load_dataset()

m,n_h,n_w=X_train.shape

permutation = list(np.random.permutation(m))
X_train = X_train[permutation,:]
Y_train = Y_train[permutation,:]
model = model(input_shape = X_train[0].shape)

model.summary()
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.001) #hyperparameters
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

train_losses = []
train_accs = []
dev_losses = []
dev_accs=[]
dev_aurocs=[]

for i in range(50):
    print("Epoch", i)
    history = model.fit(X_train, Y_train, batch_size = 64, epochs=1)
    loss, acc = model.evaluate(X_test, Y_test)
    Y_score = model.predict(X_test)
    print ("Dev set loss = ", loss)
    dev_losses.append(loss)
    print("Dev set accuracy = ", acc)
    dev_accs.append(acc)
    auroc=roc_auc_score(Y_test, Y_score)
    print ("roc_auc_score = ", auroc)
    dev_aurocs.append(auroc)
    train_losses.append(history.history['loss'])
    train_accs.append(history.history['acc'])

stats = {}
stats['train_losses'] = train_losses
stats['train_accs'] = train_accs
stats['dev_losses'] = dev_losses
stats['dev_accs'] = dev_accs
stats['dev_aurocs'] = dev_aurocs

np.save('model_2_fft_stats.npy', stats)
