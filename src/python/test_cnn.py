from __future__ import print_function

import sys,os
import theano
import theano.tensor as T
import lasagne
import numpy as np

# -----------------------------------------------------------------------------
# Data set

"""
load_dataset() is included in the mnist.py example of Lasagne at
https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
"""

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset()

#X_train.shape
#> (50000, 1, 28, 28)
#
# X_train[example][channel][row][column]
# This is what the first example looks like:
#
#array([[[ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        ...,
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.]]], dtype=float32)

#y_train.shape
#> (50000,)
# Just an array

# -----------------------------------------------------------------------------
# CNN - network structure

n_examples = None
n_channels = X_train.shape[1] #1
width  = X_train.shape[2] #28
height = X_train.shape[3] #28
n_classes = len(np.unique(y_train)) #10

# Input layer
l_in = lasagne.layers.InputLayer(
        shape=(n_examples, n_channels, width, height))

# Convolutional layer
# (ReLU is the common nonlinearity in cnns)
l_conv = lasagne.layers.Conv2DLayer(
        l_in, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)

# Pooling layer
l_pool = lasagne.layers.MaxPool2DLayer(
        l_conv, pool_size=(2, 2))

# (You can add more conv+pool layers)

# Output layer
l_output = lasagne.layers.DenseLayer(
    l_pool, num_units=n_classes,
    nonlinearity=lasagne.nonlinearities.softmax)

# Put layers together & get output
net_output = lasagne.layers.get_output(l_output)

# -----------------------------------------------------------------------------
# Define the learning

# Objective
true_output = T.ivector('true_output')
loss = T.mean(lasagne.objectives.categorical_crossentropy(net_output, true_output))

# Update
all_params = lasagne.layers.get_all_params(l_output)
updates = lasagne.updates.adadelta(loss, all_params)

# Train function
train = theano.function([l_in.input_var, true_output], loss, updates=updates)

# -----------------------------------------------------------------------------
# Define the classification

# Compute the network's output given an input
get_output = theano.function([l_in.input_var],
                             lasagne.layers.get_output(l_output, deterministic=True))

# -----------------------------------------------------------------------------
# Training

"""
Code for training is taken from the Lasagne tutorial by craffel@github
http://nbviewer.jupyter.org/github/craffel/Lasagne-tutorial/blob/master/examples/tutorial.ipynb
"""

BATCH_SIZE = 100
N_EPOCHS = 10
batch_idx = 0
epoch = 0

dataset = {
    'train': {'X': X_train, 'y': y_train},
    'valid': {'X': X_valid, 'y': y_valid}}

while epoch < N_EPOCHS:
    # Training
    #
    # dataset['train']['X'] is shape (50000, 1, 28, 28)
    # dataset['train']['X'][batch_idx:batch_idx + BATCH_SIZE] for num examples
    train(dataset['train']['X'][batch_idx:batch_idx + BATCH_SIZE],
          dataset['train']['y'][batch_idx:batch_idx + BATCH_SIZE])
    batch_idx += BATCH_SIZE

    # Once we've trained on the entire training set...
    if batch_idx >= dataset['train']['X'].shape[0]:
        # Reset the batch index & update the number of epochs trained
        batch_idx = 0
        epoch += 1

        # Get prediction on validation set
        val_output = get_output(dataset['valid']['X'])
        val_predictions = np.argmax(val_output, axis=1)
        # The accuracy is the average number of correct predictions
        accuracy = np.mean(val_predictions == dataset['valid']['y'])
        print("Epoch {} validation accuracy: {}".format(epoch, accuracy))
