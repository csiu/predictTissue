from __future__ import print_function

import argparse, os
import theano
import theano.tensor as T
import lasagne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Define parser
def getargs():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-m', '--mode', choices=['inspect', 'predict'],
                        default='inspect')
    parser.add_argument('-i', '--indir', default="testdata")

    args = parser.parse_args()
    return args

args = getargs()

# Prep input
if args.mode == 'inspect':
    df = pd.read_csv(os.path.join(args.indir, "train.csv"))
    df_X = df.iloc[:,1:]
    df_y = df.iloc[:,0]
    X, X_test, y, y_test = train_test_split(df_X, df_y, test_size=0.8)
else: 
    df = pd.read_csv(os.path.join(args.indir, "train.csv"))
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    X_new = pd.read_csv(os.path.join(args.indir, "test.csv"))

# Define variables
N_CLASSES = len(y.unique())
N_UNITS = int((X.shape[1] + N_CLASSES)/2)
N_EPOCHS = 1000
LEARNING_RATE = 0.001

# Define layer structure
l_in = lasagne.layers.InputLayer(shape=X.shape)
l_hidden = lasagne.layers.DenseLayer(
    l_in, num_units=N_UNITS, nonlinearity=lasagne.nonlinearities.sigmoid)
l_output = lasagne.layers.DenseLayer(
    l_hidden, num_units=N_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
net_output = lasagne.layers.get_output(l_output)

# Define objective
true_output = T.ivector('true_output')
loss = T.mean(lasagne.objectives.categorical_crossentropy(net_output, true_output))

# Define update
all_params = lasagne.layers.get_all_params(l_output)
updates = lasagne.updates.adam(loss, all_params, learning_rate=LEARNING_RATE)
train = theano.function([l_in.input_var, true_output], loss, updates=updates)
get_output = theano.function([l_in.input_var], net_output)

# Train for N_EPOCHS
for n in range(N_EPOCHS):
    print(n, train(X, y))

if args.mode == 'inspect':
    # Evaluation on training data & validation data
    y_predicted = np.argmax(get_output(X), axis=1)
    print(metrics.accuracy_score(y, y_predicted))
    print(metrics.accuracy_score(y_test, np.argmax(get_output(X_test), axis=1)))
else:
    OUTFILE = "day27-node{}-learn{}-epoch{}.csv".format(
            N_UNITS, LEARNING_RATE, N_EPOCHS)
    # Make predictions using trained model
    y_new = np.argmax(get_output(X_new), axis=1)
    y_new = pd.DataFrame(y_new, columns=['Label'])
    y_new.insert(0, 'ImageId', range(1, len(y_new)+1))
    y_new.to_csv(OUTFILE, index=False)
