from __future__ import print_function

import argparse, os
import theano
import theano.tensor as T
import lasagne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

import cPickle as pickle

class PickleModel:
    def __init__(self, network, model_file):
        self.network = network
        self.model_file = model_file

    def save_model(self):
        net_info = {'network': self.network,
                   'params': lasagne.layers.get_all_param_values(self.network)}
        pickle.dump(net_info, open(self.model_file, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        net = pickle.load(open(self.model_file, 'rb'))
        all_params = net['params']
        lasagne.layers.set_all_param_values(self.network, all_params)

def getargs():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-i', '--indir', default="testdata")
    parser.add_argument('-o', '--outprefix', default="out")

    parser.add_argument('-n', '--hidden', type=int, required=True,
                        help='Number of hidden nodes')
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='Number of validation epochs (iterations)')
    parser.add_argument('-l', '--learn', type=float, default=0.001,
                        help='Learning rate')

    parser.add_argument('-m', '--model',
                        help='Model file')

    parser.add_argument('mode', choices=['train', 'test'])

    args = parser.parse_args()
    return args

class LoadCustomTissueInput():
    """
    To load (X, y) from custom input_file and class_file.

    input_file is a CSV file with samples as columns and features as rows.
    There are column names (ie. sample_ids) but no feature/row names.

    class_file is a TSV file mapping sample_ids of the input_file (column 1) to
    the tissue labels (column 2).
    """
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
        y_new = pd.DataFrame(y_predicted, columns=['Label'])
        y_new.insert(0, 'ImageId', range(1, len(y_new)+1))
        y_new.to_csv(out_file, index=False)

def build_mlp(shape, num_units, num_classes):
    l_in = lasagne.layers.InputLayer(shape=shape)

    l_hidden = lasagne.layers.DenseLayer(
            l_in, num_units=num_units,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    l_output = lasagne.layers.DenseLayer(
            l_hidden, num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax)

    return(l_output, l_in.input_var)

if __name__ == '__main__':
    args = getargs()

    N_UNITS = args.hidden
    N_EPOCHS = args.epochs
    LEARNING_RATE = args.learn
    OUT_PREFIX = args.outprefix

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
        X, y, X_new = td.load_test()

    # Prepare Theano variables for inputs and targets
    target_var = T.ivector('target_var')

    network, input_var = build_mlp(shape=X.shape, num_units=N_UNITS, num_classes=10)

    # Objective
    net_output = lasagne.layers.get_output(network)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(
            net_output, target_var))
    # Update
    all_params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.adam(loss, all_params,
                                   learning_rate=LEARNING_RATE)
    # Train
    train = theano.function([input_var, target_var], loss,
                            updates=updates)

    # Training
    for n in range(N_EPOCHS):
        print(n, train(X, y))

    # Predict
    get_output = theano.function([input_var], net_output)

    # Evaluation
    print(metrics.accuracy_score(y, np.argmax(get_output(X), axis=1)))
    if(args.mode == 'train'):
        print(metrics.accuracy_score(y_test, np.argmax(get_output(X_test), axis=1)))
    else:
        # Make predictions using trained model
        out_file = "{}-node{}-learn{}-epoch{}.csv".format(
                OUT_PREFIX, N_UNITS, LEARNING_RATE, N_EPOCHS)
        td.write_submission(np.argmax(get_output(X_new), axis=1), out_file)
