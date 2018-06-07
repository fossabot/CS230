#

# -*- coding: utf-8 -*-
import numpy as np
#import tensorflow as tf
import math
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv2D, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
    ### END CODE HERE ##

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
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(16, 10, strides=10)(X_input)                               # CONV1D
    X = BatchNormalization()(X)                          # Batch normalization
    # X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.3)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = LSTM(units = 64, return_sequences = True)(X)                            # GRU (use 128 units and return the sequences)
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Dropout(0.3)(X)                               # dropout (use 0.8)

    # Step 3: Second GRU Layer (≈4 lines)
    X = LSTM(units = 128, return_sequences = False)(X)                       # GRU (use 128 units and return the sequences)
    #X = BatchNormalization()(X)                                 # Batch normalization
    X = Dropout(0.3)(X)                                 # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = Dense(1, activation = "sigmoid")(X) # time distributed  (sigmoid)
    #X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    ### END CODE HERE ###

    model = Model(inputs = X_input, outputs = X)

    return model

X_train, Y_train, X_test, Y_test = load_dataset()

print(X_train.shape)
m,n_h,n_w=X_train.shape

permutation = list(np.random.permutation(m))
X_train = X_train[permutation,:]
Y_train = Y_train[permutation,:]
print(Y_train)
model = model(input_shape = X_train[0].shape)

model.summary()
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

for i in range(30):
    print("Epoch", i)
    model.fit(X_train, Y_train, batch_size = 64, epochs=1)
    loss, acc = model.evaluate(X_test, Y_test)
    print ("Dev set loss = ", loss)
    print("Dev set accuracy = ", acc)
