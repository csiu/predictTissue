from __future__ import print_function

import argparse, os
import theano
import theano.tensor as T
import lasagne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

def getargs():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-i', '--indir', default="testdata")
    parser.add_argument('-n', '--hidden', type=int, required=True,
                        help='Number of hidden nodes')
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='Number of validation epochs (iterations)')
    parser.add_argument('-l', '--learn', type=float, default=0.001,
                        help='Learning rate')

    parser.add_argument('mode', choices=['train', 'test'])
    
    args = parser.parse_args()
    return args

class BasicMLP:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.shape = X.shape
        self.num_classes = len(y.unique())

    def network(self, num_units, learning_rate):
        # Define layer structure
        self.l_in = lasagne.layers.InputLayer(shape=self.shape)
        l_hidden = lasagne.layers.DenseLayer(
                self.l_in, num_units=num_units,
                nonlinearity=lasagne.nonlinearities.sigmoid)
        l_output = lasagne.layers.DenseLayer(
                l_hidden, num_units=self.num_classes,
                nonlinearity=lasagne.nonlinearities.softmax)
        self.net_output = lasagne.layers.get_output(l_output)

        # Define objective
        true_output = T.ivector('true_output')
        loss = T.mean(lasagne.objectives.categorical_crossentropy(
                self.net_output, true_output))

        # Define update
        all_params = lasagne.layers.get_all_params(l_output)
        updates = lasagne.updates.adam(loss, all_params,
                                       learning_rate=learning_rate)
        self.train = theano.function([self.l_in.input_var, true_output], loss,
                                     updates=updates)

    def train_network(self, n_epochs):
        for n in range(n_epochs):
            print(n, self.train(self.X, self.y))

    def get_output(self, X2):
        get_output = theano.function([self.l_in.input_var], self.net_output)
        return(get_output(X2))

args = getargs()

# Prep input
if(args.mode == 'train'):
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
N_UNITS = args.hidden
N_EPOCHS = args.epochs
LEARNING_RATE = args.learn

bmlp = BasicMLP(X, y)
bmlp.network(N_UNITS, LEARNING_RATE)
bmlp.train_network(N_EPOCHS)

if(args.mode == 'train'):
    # Evaluation on training data & validation data
    y_predicted = np.argmax(bmlp.get_output(X), axis=1)
    print(metrics.accuracy_score(y, y_predicted))
    print(metrics.accuracy_score(y_test, np.argmax(bmlp.get_output(X_test), axis=1)))
else:
    OUTFILE = "day27-node{}-learn{}-epoch{}.csv".format(
            N_UNITS, LEARNING_RATE, N_EPOCHS)
    # Make predictions using trained model
    y_new = np.argmax(bmlp.get_output(X_new), axis=1)
    y_new = pd.DataFrame(y_new, columns=['Label'])
    y_new.insert(0, 'ImageId', range(1, len(y_new)+1))
    y_new.to_csv(OUTFILE, index=False)
