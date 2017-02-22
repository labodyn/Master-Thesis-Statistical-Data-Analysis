#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import theano as th
import theano.tensor as T
import numpy as np
import time
import pickle
from six.moves import cPickle
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cross_validation import cross_val_score

# Import from own modules
from functions import init_weight, init_bias
from plot import knn_par_plot

def train_knn(data_train, data_test, regions_train, regions_test, model_name, 
        n_neighbors=None):
    """ Build a k nearest neighbor classifier and return the performance on 
    the test set """

    # Make numpy array from data
    x_train = np.array(data_train)
    y_train = np.array(regions_train)
    x_test = np.array(data_test)
    y_test = np.array(regions_test)

    # Cross Validation for hyperparameters
    if n_neighbors is None:
        print('Determining knn n_neighbors with 5fold cross-validation.')
        neighbors_list = [1, 5, 10, 12, 15, 18, 20, 25, 30, 50]
        accuracy_list = []
        for n_neighbors in neighbors_list:
            print('CrossValidating knn with {} neighbors'.format(n_neighbors))
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            cv_accuracy = sum(cross_val_score(clf, x_train, y_train, cv=5))/5
            accuracy_list.append(cv_accuracy)
        knn_par_plot(neighbors_list, accuracy_list, model_name)
        n_neighbors = neighbors_list[accuracy_list.index(max(accuracy_list))]

    # Fit model with highest CV accuracy
    clf_opt = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf_opt.fit(x_train, y_train)

    # Test performance
    y_predict = clf_opt.predict(x_test)
    print('KNN accuracy: {:1.4f}'.format(accuracy_score(y_test, y_predict)))
    print('KNN confusion Matrix:\n', confusion_matrix(y_test, y_predict))

    return y_predict

def train_qda(data_train, data_test, regions_train, regions_test):
    """ Build a Quadratic discriminant analysis classifier and return the 
    performance on the test set """

    # Build Model
    x_train = np.array(data_train)
    y_train = np.array(regions_train)
    x_test = np.array(data_test)
    y_test = np.array(regions_test)
    clf = QDA()
    clf.fit(x_train, y_train)

    # Test performance
    y_predict = clf.predict(x_test)
    print('QDA accuracy: {:1.4f}'.format(accuracy_score(y_test, y_predict)))
    print('QDA confusion Matrix:\n', confusion_matrix(y_test, y_predict))

    return y_predict

def train_zero_prediction(x, p):
    """ Calculate the loss for predicting zero for every value. """

    if p.loss_fn.__name__ == 'least_squares':
        loss = np.mean((x)**2)
    elif p.loss_fn.__name__ == 'cross_entropy':
        lossmatrix = -x*np.log(0)
        lossmatrix[np.isnan(lossmatrix)] = 0
        loss = np.mean(lossmatrix)
    return loss

def train_svd(x_train, x_val, p):
    """ Peform singular value decomposition on the data.  Reduce to 'dim' 
    number of dimensions.  Calculate the loss the dimension reduction has. """

    # Singular value decomposition
    print('Performing singular value decomposition...')
    dim = min(p.n_hidden_neurons)
    U, D, VT = np.linalg.svd(x_train, full_matrices=False)
    V_k = VT.T[:,0:dim]
    z_k = np.dot(x_val, V_k)
    x_k = np.dot(z_k, V_k.T)

    # Calculate loss
    if p.loss_fn.__name__ == 'least_squares':
        loss = np.mean((x_k-x_val)**2)
    elif p.loss_fn.__name__ == 'cross_entropy':
        loss_matrix = -(x_val*np.log(x_k) + (1 - x_val)*np.log(1 - x_k))
        loss_matrix[np.isnan(loss_matrix)] = 0
        loss = np.mean(loss_matrix)

    return loss, V_k
