#

# -*- coding: utf-8 -*-
import numpy as np
#import tensorflow as tf
import math
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv2D, Conv1D, Flatten
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
    ### END CODE HERE ##

def load_dataset():
    X_train = np.load('Xtrain4.npy')
    Y_train = np.load('Ytrain4.npy')
    X_test = np.load('Xtest4.npy')
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
    X = Conv1D(10, 100, strides=10)(X_input)                               # CONV1D
    X = BatchNormalization()(X)                          # Batch normalization
    X = Flatten()(X)
    X = Dropout(0.7)(X)
    # X = Activation('relu')(X)                                 # ReLu activation
    X = Dense(2000, activation = "relu")(X)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = Dense(500, activation = "relu")(X)
    X = BatchNormalization()(X)                          # Batch normalization
    X = Dropout(0.7)(X)
    X = Dense(100, activation = "relu")(X)
    X = BatchNormalization()(X)                          # Batch normalization
    X = Dropout(0.4)(X)
    X = Dense(1, activation = "sigmoid")(X)
    #X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    ### END CODE HERE ###

    model = Model(inputs = X_input, outputs = X)

    return model

X_train, Y_train, X_test, Y_test = load_dataset()

m,n_h,n_w=X_train.shape

permutation = list(np.random.permutation(m))
X_train = X_train[permutation,:]
Y_train = Y_train[permutation,:]

print(Y_train)
print(X_train[1].shape)

model = model(input_shape = X_train[0].shape)

model.summary()
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

loss, acc = model.evaluate(X_test, Y_test)
print ("Dev set loss = ", loss)
print("Dev set accuracy = ", acc)

for i in range(40):
    print("Epoch", i)
    model.fit(X_train, Y_train, batch_size = 64, epochs=1)
    loss, acc = model.evaluate(X_test, Y_test)
    print ("Dev set loss = ", loss)
    print("Dev set accuracy = ", acc)
