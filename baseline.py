import numpy as np
import tensorflow as tf
import math
from tensorflow.python.framework import ops

def load_dataset():
    X_train = np.load('Xtrain.npy')
    Y_train = np.load('Ytrain.npy')
    #Seq_train = np.load('Seqtrain.npy')
    X_test = np.load('Xtest.npy')
    Y_test = np.load('Ytest.npy')
    #Seq_test = np.load('Seqtest.npy')
    return X_train, Y_train, X_test, Y_test

def create_placeholders(n_H0, n_W0, n_y, n_C):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X =  tf.placeholder(tf.float32, shape=(None,n_H0,n_W0,n_C))
    Y =  tf.placeholder(tf.float32, shape=(None,n_y))

    return X, Y

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)                              # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [16,100,1,1], initializer =tf.contrib.layers.xavier_initializer(seed = 0) )
    parameters = {"W1": W1}
    return parameters

def forward_propagation(X, parameters):
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

    # CONV2D: stride of 10 in one dimension, padding 'valid'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,10,1,1], padding = 'VALID')
    # RELU
    A1 = tf.nn.relu(Z1)
    # FLATTEN
    A1 = tf.contrib.layers.flatten(A1)
    #A1 = tf.concat([Z1,tf.transpose(Seq)],0)
    # FULLY-CONNECTED layers
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    A2 = tf.contrib.layers.fully_connected(A1, 100)
    A3 = tf.contrib.layers.fully_connected(A2, 20)
    Z4 = tf.contrib.layers.fully_connected(A3, 1, activation_fn=None)

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

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z4, labels = Y))
    return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[0]
                     # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[ k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch_Y = shuffled_Y[ k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:m,:]
        mini_batch_Y =shuffled_Y[num_complete_minibatches*mini_batch_size:m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 1, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D ->  FLATTEN -> add SEQ data -> 3x FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 16, 11988, 1)
    Y_train -- test set, of shape (None, n_y = 1)
    X_test -- training set, of shape (None, 16, 11988, 1)
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
    (m, n_H0, n_W0) = X_train.shape
    X_train = X_train.reshape(m,n_H0,n_W0,1)
    n_y = Y_train.shape[1]

    (m1, _, _) = X_test.shape
    X_test = X_test.reshape(m1, n_H0, n_W0, 1)
    costs = []                                        # To keep track of the cost

    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_y, 1)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z4 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z4,Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

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
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # Calculate the correct predictions
        # def fn_true(): return tf.constant(1, dtype = tf.float32)
        # def fn_false(): return tf.constant(0, dtype = tf.float32)
        # predict_op = tf.cond(tf.greater(tf.squeeze(Z4), tf.constant(0.5)), true_fn = fn_true, false_fn = fn_false)
        # Z4 = tf.squeeze(Z4)
        # comp = tf.Variable(tf.fill(dims=tf.shape(Z4), 0.5), validate_shape=False)
        # predict_op = tf.greater(Z4, comp)
        # correct_prediction = tf.equal(tf.cast(predict_op, dtype = tf.float32), Y)
        correct_prediction = tf.round(tf.squeeze(Z4))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("here1")
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        print("here2")
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("here3")
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print("Parameters:", parameters)

        return train_accuracy, test_accuracy, parameters

X_train, Y_train, X_test, Y_test = load_dataset()
_, _, parameters = model(X_train, Y_train, X_test, Y_test)
