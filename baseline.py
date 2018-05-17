import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)

class NeuralData:
    def __init__(self, dataType, dog, trial, sequenceNum, samplingRate, dataArray):
        self.dataType = dataType #preictal or interictal
        self.dog = dog
        self.trial = trial
        self.sequenceNum = sequenceNum
        self.samplingRate = samplingRate
        self.dataArray = dataArray


def load_dataset():
	X_train = np.load('Xtrain.npy')
	Y_train = np.load('Ytrain.npy')
	Seq_train = np.load('Seqtrain.npy')
	X_test = np.load('Xtest.npy')
	Y_test = np.load('Ytest.npy')
	Seq_test = np.load('Seqtest.npy')
	return X_train, Y_train, Seq_train, X_test, Y_test, Seq_test

X_train, Y_train, Seq_train, X_test, Y_test, Seq_test= load_dataset()


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X =  tf.placeholder(tf.float32, shape=(None,n_H0,n_W0,n_C0))
    Y =  tf.placeholder(tf.float32, shape=(None,n_y))
    Seq =  tf.placeholder(tf.float32, shape=(None,n_y))

    return X, Y, Seq

X, Y, Seq = create_placeholders(16, 12000, 1, 1)

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)                              # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.get_variable("W1", [16,100,1,1], initializer =tf.contrib.layers.xavier_initializer(seed = 0) )
    #W2 =  tf.get_variable("W2", [2,2,8,16], initializer =tf.contrib.layers.xavier_initializer(seed = 0))
    ### END CODE HERE ###

    parameters = {"W1": W1}

    return parameters

def forward_propagation(X, parameters, Seq):
    """
    Implements the forward propagation for the model:
    CONV2D ->  FLATTEN -> add SEQ data -> 3x FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    #W2 = parameters['W2']

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,10,1,1], padding = 'VALID')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    #P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    #Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    #A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    #P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    # FLATTEN
    A1 = tf.contrib.layers.flatten(A1)
    A1 = tf.concat([Z1,tf.transpose(Seq)],0)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z2 = tf.contrib.layers.fully_connected(A1, 100, activation_fn=relu)
    Z3 = tf.contrib.layers.fully_connected(Z2, 20, activation_fn=relu)
    Z4 = tf.contrib.layers.fully_connected(Z3, 1, activation_fn=None)

    ### END CODE HERE ###

    return Z4

def compute_cost(Z4, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z4, labels = Y))
    ### END CODE HERE ###

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D ->  FLATTEN -> add SEQ data -> 3x FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 16, 12000, 1)
    Y_train -- test set, of shape (None, n_y = 1)
    X_test -- training set, of shape (None, 16,12000, 1)
    Y_test -- test set, of shape (None, n_y = 1)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []                                        # To keep track of the cost

    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y, Seq = create_placeholders(n_H0, n_W0, n_C0, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z4 = forward_propagation(X, parameters, Seq)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z4,Y)
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###

                minibatch_cost += temp_cost / num_minibatches


            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)


        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # Calculate the correct predictions
        if tf.greater(Z4,tf.variable(0.5)):
        	predict_op = 1
        else:
        	predict_op = 0
        correct_prediction = tf.equal(predict_op, Y)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train, Seq: Seq_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test, Seq: Seq_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters
