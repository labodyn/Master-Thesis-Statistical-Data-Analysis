#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import theano as th
import theano.tensor as T
import numpy as np
#import math
#import pandas as pd
import time
#import random as r
from sklearn.cross_validation import train_test_split

from functions import *

class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self


def predict_zero(x, costfn):

    '''Calculate the costfunction for predicting zero for every value.'''

    if costfn.__name__ == 'least_squares':
        cost = np.mean((x)**2)
    elif costfn.__name__ == 'cross_entropy':
        costmatrix = -x*np.log(0)
        costmatrix[np.isnan(costmatrix)] = 0
        cost = np.mean(costmatrix)
    return cost

def perform_svd(x, dim, costfn):
    ''' Peform singular value decomposition on the data.
    Reduce to 'dim' number of dimensions.
    Calculate the cost the dimension reduction has.
    '''

    #Singular value decomposition
    print('Performing singular value decomposition...')
    U, D, VT = np.linalg.svd(x, full_matrices=False)
    V_k = VT.T[:,0:dim]
    z_k = np.dot(x, V_k)
    x_k = np.dot(z_k, V_k.T)

    #Calculate cost
    if costfn.__name__ == 'least_squares':
        cost = np.mean((x_k-x)**2)
    elif costfn.__name__ == 'cross_entropy':
        costmatrix = -(x*np.log(x_k) + (1 - x)*np.log(1 - x_k))
        costmatrix[np.isnan(costmatrix)] = 0
        cost = np.mean(costmatrix)

    return cost, z_k


def create_batches(x_train, batch_size):
    train_batches = []
    for i in range(0, len(x_train), batch_size):
        train_batches.append(x_train[i:i+batch_size])

    #Blend last two batches together if last batch doesn't have full size.
    if len(train_batches[-1]) < batch_size:
        last_batch = train_batches.pop()
        train_batches[-2] += last_batch

    return train_batches


def train_autoencoder(data, network, graddesc):
    ''' Initialize the network parameters in theano.
    Execute a loop that calculates the gradient using backward propagation.
    Use minibatches and momentum to speed up the learning.
    The algorithm stops when either max_loops, max_time or low_cost 
    has been reached. '''

    #Split in train and test data 
    data_train, data_test = train_test_split(data, 
            test_size=graddesc.test_size, random_state=42)
    x_train = data_train.tolist()
    x_test = data_test.tolist()
    print('Split data in {} training data and {} testing data'
                        .format(len(x_train), len(x_test)))

    #Create mini-batches with size 'batch_size'
    train_batches = create_batches(x_train, graddesc.batch_size)

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
    n_features = len(x_train[0])
    n_neurons = [n_features] + network.n_hidden_neurons + [n_features]
    bias_init = graddesc.eps_init if network.has_bias else 0
    for i in range(len(n_neurons) - 1):
        weights.append(init_weight(n_neurons[i], n_neurons[i+1], 
            graddesc.eps_init))
        weights_momentum.append(init_weight(n_neurons[i], 
            n_neurons[i+1], graddesc.eps_init))
        biases.append(init_bias(n_neurons[i+1], bias_init))
        biases_momentum.append(init_bias(n_neurons[i+1], bias_init))
        neurons.append(network.actfn[i](neurons[i], weights[i], biases[i]))

    #Define the cost
    cost = network.costfn(neurons[-1], target)

    #Define the gradients
    for weight, bias in zip(weights, biases):
        weights_grad.append(T.grad(cost, weight))
        biases_grad.append(T.grad(cost, bias))

    #Define updates for weights parameters
    updates = []
    for weight, weight_momentum, weight_grad in zip(weights, 
            weights_momentum, weights_grad):
        updates.append((weight, weight - graddesc.delta*weight_momentum))
        updates.append((weight_momentum, weight_momentum 
            - graddesc.alpha*(weight_momentum - weight_grad)))

    #Define updates for biases parameters
    if network.has_bias:
        for bias, bias_momentum, bias_grad in zip(biases, 
                biases_momentum, biases_grad):
            updates.append((bias, bias - graddesc.delta*bias_momentum))
            updates.append((bias_momentum, bias_momentum 
                - graddesc.alpha*(bias_momentum - bias_grad)))

    #Define usefull functions in Theano
    neuron_first_layer = neurons[0]
    neuron_bottleneck_layer = neurons[n_neurons.index(min(n_neurons))]
    perform_graddesc = th.function([neuron_first_layer, target], 
        updates=updates)
    give_cost = th.function([neuron_first_layer, target], cost)
    give_bottleneck_neurons = th.function([neuron_first_layer], 
            neuron_bottleneck_layer)

    #Define parameters for the gradient descent
    cost_train = []
    cost_test = []
    used_data = []
    start_time = time.time()
    running_time = 0
    loops_counter = 0
    converged = False

    #Perform gradient descent over all data untill converged
    print('Training the network...')
    while not converged:

        #Number of loops
        loops_counter += 1

        #Loop over minibatches
        for train_batch in train_batches:

            #Do a gradient descent step with the mini-batch.
            perform_graddesc(train_batch,train_batch)

            #Store the data that is used in the current cost step 
            used_data += train_batch

            #If sufficient data has been used, update the cost lists.
            if len(used_data) > graddesc.cost_update_size:

                #Update cost_train and cost_test
                cost_train.append(float(give_cost(used_data, used_data)))
                cost_test.append(float(give_cost(x_test, x_test)))
                print(('Data loops: {:2}, Running time: {:4.0f}s, '
                        'Train cost: {:1.4f}, Test cost: {:1.4f}')
                        .format(loops_counter, running_time, cost_train[-1],
                        cost_test[-1]))

                #Reset the list of training data in the current cost step
                used_data = []

                #Stopping criteria
                running_time = time.time() - start_time
                converged = (cost_test[-1] < graddesc.low_cost 
                    or loops_counter > graddesc.max_loops
                    or running_time > graddesc.max_time)

                if converged:
                    print('Gradient descent has converged!')
                    break

    #Calculate variables to return
    bottleneck_neurons = give_bottleneck_neurons(data.tolist())
    return bottleneck_neurons, cost_train, cost_test, running_time
