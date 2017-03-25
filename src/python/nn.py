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


class LoadCustomTissueInput():
    def __init__(self, input_file, class_file):
        self.input_file = input_file
        self.class_file = class_file

    def _load_x(self, do_transpose=True):
        self.X = pd.read_csv(self.input_file, index_col=False)
        if do_transpose: self.X = self.X.transpose()
        return(self.X)

    def _load_y(self):
        df = pd.read_table(self.class_file)
        class_lookup = pd.Series(df['label'].values, index=df['sample']).to_dict()
        y = pd.Series([class_lookup[s] for s in self.X.index])
        return(y)

    def load_data(self):
        # Class label needs to be numerical
        class_mapping = {
                'Blood':0,'Brain':1, 'Breast':2, 'Colon':3, 'Thyroid':4}
        X = self._load_x().reset_index(drop=True)
        y = self._load_y().map(class_mapping)
        return(X, y)

class ToyData():
    """
    Load Kaggle Digit Recognizer MNIST data from datadir &
    write predictions in the Kaggle submission format
    """
    def __init__(self, datadir):
        self.datadir = datadir

    def _load_train(self):
        df = pd.read_csv(os.path.join(self.datadir, "train.csv"))
        X = df.iloc[:,1:]
        y = df.iloc[:,0]
        return(X, y)

    def load_train(self):
        df_X, df_y = self._load_train()
        X, X_test, y, y_test = train_test_split(df_X, df_y, test_size=0.8)
        return(X, X_test, y, y_test)

    def load_test(self):
        X, y = self._load_train()
        X_new = pd.read_csv(os.path.join(self.datadir, "test.csv"))
        return(X, y, X_new)
    
    def write_submission(self, y_predicted, out_file):
        y_new = np.argmax(y_predicted, axis=1)
        y_new = pd.DataFrame(y_new, columns=['Label'])
        y_new.insert(0, 'ImageId', range(1, len(y_new)+1))
        y_new.to_csv(out_file, index=False)


args = getargs()

# Prep input
td = ToyData(args.indir)
if(args.mode == 'train'):
    X, X_test, y, y_test = td.load_train()

    #proj_dir = "/projects/csiu_prj_results/PROJECTS/predictTissue"
    #input_file = os.path.join("results/features/promoter/input.txt")
    #class_file = os.path.join("metadata/tissueclass.txt")
    #df_X, df_y = LoadCustomTissueInput(input_file, class_file).load_data()
    #X, X_test, y, y_test = train_test_split(df_X, df_y, test_size=0.8)
else:
# Define variables
N_UNITS = args.hidden
N_EPOCHS = args.epochs
LEARNING_RATE = args.learn
    X, y, X_new = td.load_test()

bmlp = BasicMLP(X, y)
bmlp.network(N_UNITS, LEARNING_RATE)
bmlp.train_network(N_EPOCHS)

if(args.mode == 'train'):
    # Evaluation on training data & validation data
    y_predicted = np.argmax(bmlp.get_output(X), axis=1)
    print(metrics.accuracy_score(y, y_predicted))
    print(metrics.accuracy_score(y_test, np.argmax(bmlp.get_output(X_test), axis=1)))
else:
    # Make predictions using trained model
    OUTFILE = "day27-node{}-learn{}-epoch{}.csv".format(
    N_UNITS, LEARNING_RATE, N_EPOCHS)
    td.write_submission(y_predicted=bmlp.get_output(X_new), OUTFILE)
