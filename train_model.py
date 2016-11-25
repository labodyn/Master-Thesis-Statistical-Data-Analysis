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
from sklearn.metrics import accuracy_score, confusion_matrix

from functions import *


def performance_knn():
    pass

def performance_lda(data_train, data_test, regions_train, regions_test):
    '''Build a linear discriminant analysis classifier.
    return the performance on the test set'''

    #Build Model
    x_train = np.array(data_train)
    y_train = np.array(regions_train)
    x_test = np.array(data_test)
    y_test = np.array(regions_test)
    clf = QDA()
    clf.fit(x_train, y_train)

    #Test performance
    y_predict = clf.predict(x_test)
    print('Accuracy: {:1.4f}'.format(accuracy_score(y_test, y_predict)))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_predict))

    return y_predict

def perform_zero_prediction(x, p):
    '''Calculate the cost for predicting zero for every value.'''

    if p.cost_fn.__name__ == 'least_squares':
        cost = np.mean((x)**2)
    elif p.cost_fn.__name__ == 'cross_entropy':
        costmatrix = -x*np.log(0)
        costmatrix[np.isnan(costmatrix)] = 0
        cost = np.mean(costmatrix)
    return cost


def perform_svd(x, p):
    ''' Peform singular value decomposition on the data.
    Reduce to 'dim' number of dimensions.
    Calculate the cost the dimension reduction has.'''

    #Singular value decomposition
    print('Performing singular value decomposition...')
    dim = min(p.n_hidden_neurons)
    U, D, VT = np.linalg.svd(x, full_matrices=False)
    V_k = VT.T[:,0:dim]
    z_k = np.dot(x, V_k)
    x_k = np.dot(z_k, V_k.T)

    #Calculate cost
    if p.cost_fn.__name__ == 'least_squares':
        cost = np.mean((x_k-x)**2)
    elif p.cost_fn.__name__ == 'cross_entropy':
        cost_matrix = -(x*np.log(x_k) + (1 - x)*np.log(1 - x_k))
        cost_matrix[np.isnan(cost_matrix)] = 0
        cost = np.mean(cost_matrix)

    return cost, z_k

def perform_autoencoding(data, p):
    ''' Initialize the autoencoder network in theano.
    Execute a loop that calculates the gradient using backward propagation.
    Use minibatches and momentum to speed up the learning.
    The algorithm stops when either max_loops, max_time or low_cost 
    has been reached.  Return a list with train and test cost, 
    along with run time and the value of the bottleneck neurons.
    Write the model to file'''

    #Split in train and test data 
    data_train, data_test = train_test_split(data, 
            test_size=p.test_size, random_state=42)
    x_train = data_train.tolist()
    x_test = data_test.tolist()
    print('Split data in {} training data and {} testing data'
                        .format(len(x_train), len(x_test)))

    #Create minibatches of size 'batchsize'. 
    train_batches = []
    for i in range(0, len(x_train), p.batch_size):
        train_batches.append(x_train[i:i + p.batch_size])
    #Blend last two batches together if last batch doesn't have full size.
    if len(train_batches[-1]) < p.batch_size:
        last_batch = train_batches.pop()
        train_batches[-2] += last_batch

    #Define arrays over the layers of the theano variables.
    target = T.dmatrix()
    neurons = [T.dmatrix()]
    weights = []
    biases = []
    weights_grad = []
    biases_grad = []
    weights_momentum = []
    biases_momentum = []

    #Initialise all the network parameters 
    #and define the neuron values in theano
    assert len(p.n_hidden_neurons) + 1 == len(p.activation_fn)
    n_features = len(x_train[0])
    n_neurons = [n_features] + p.n_hidden_neurons + [n_features]
    bias_init = p.eps_init if p.has_bias else 0
    for i in range(len(n_neurons) - 1):
        weights.append(init_weight(n_neurons[i], n_neurons[i+1], p.eps_init))
        weights_momentum.append(init_weight(n_neurons[i], 
            n_neurons[i+1], p.eps_init))
        biases.append(init_bias(n_neurons[i+1], bias_init))
        biases_momentum.append(init_bias(n_neurons[i+1], bias_init))
        neurons.append(p.activation_fn[i](neurons[i], weights[i], biases[i]))

    #Define the cost
    cost = p.cost_fn(neurons[-1], target)

    #Define the gradients
    for weight, bias in zip(weights, biases):
        weights_grad.append(T.grad(cost, weight))
        biases_grad.append(T.grad(cost, bias))

    #Define updates for weights parameters
    updates = []
    updates_momentum = []
    for weight, weight_momentum, weight_grad in zip(weights, 
            weights_momentum, weights_grad):
        updates.append((weight, weight + weight_momentum))
        updates_momentum.append((weight_momentum, p.alpha*weight_momentum 
            - p.delta*weight_grad))

    #Define updates for biases parameters
    if p.has_bias:
        for bias, bias_momentum, bias_grad in zip(biases, 
                biases_momentum, biases_grad):
            updates.append((bias, bias + bias_momentum))
            updates_momentum.append((bias_momentum, p.alpha*bias_momentum 
                - p.delta*bias_grad))

    #Define usefull functions in Theano
    neuron_first_layer = neurons[0]
    neuron_last_layer = neurons[-1]
    neuron_bottleneck_layer = neurons[n_neurons.index(min(n_neurons))]
    give_cost = th.function([neuron_first_layer, target], cost)
    do_update = th.function([], updates=updates)
    do_update_momentum = th.function([neuron_first_layer, target], 
        updates=updates_momentum)
    give_bottleneck_layer = th.function([neuron_first_layer], 
            neuron_bottleneck_layer)
    give_last_layer = th.function([neuron_first_layer], neuron_last_layer)

    #Define parameters for the gradient descent
    cost_train = []
    cost_test = []
    used_data = []
    start_time = time.time()
    running_time = 0
    loops_counter = 0
    converged = False

    #Perform gradient descent over all data until converged or interrupted.
    print('Training the network...')
    try:
        while not converged:

            #Number of loops
            loops_counter += 1

            #Loop over minibatches
            for train_batch in train_batches:

                #Do a gradient descent step with the mini-batch.
                #Ilya Sutskever 2012. First update weights, then momentum
                do_update()
                do_update_momentum(train_batch,train_batch)

                #Store the data that is used in the current cost step
                used_data += train_batch

                #If sufficient data has been used, update the cost lists.
                if len(used_data) > p.cost_update_size:

                    #Update cost_train and cost_test
                    cost_train.append(float(give_cost(used_data, used_data)))
                    cost_test.append(float(give_cost(x_test, x_test)))
                    used_data = []
                    print(('Data loops: {:3}, Run time: {:5.0f}s, '
                            'Train cost: {:1.5f}, Test cost: {:1.5f}')
                            .format(loops_counter, running_time, 
                            cost_train[-1], cost_test[-1]))

                    #Stopping criteria
                    running_time = time.time() - start_time
                    converged = (cost_test[-1] < p.low_cost 
                        or loops_counter > p.max_loops 
                        or running_time > p.max_time)
                    if converged:
                        print('Gradient descent has converged!')
                        break

    #When converged of in case gradient descent gets interrupted
    finally:

        #Save network
        f = open('output/{}_{:0.4f}.pkl'.format(p.string, cost_test[-1]), 'wb')
        cPickle.dump(give_last_layer, f, protocol=cPickle.HIGHEST_PROTOCOL)

        bottleneck_neurons = give_bottleneck_layer(data.tolist())
        return bottleneck_neurons, cost_train, cost_test, running_time
