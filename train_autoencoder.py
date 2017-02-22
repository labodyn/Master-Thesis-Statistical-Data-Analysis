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
import os
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
from performance import make_recipes

def train_autoencoder(data_train, data_val, p, pretrained=None, fast=False):
    """ Initialize the autoencoder network in theano. Execute a loop that 
    calculates the gradient using backward propagation. Use minibatches and 
    momentum to speed up the learning. The algorithm stops when either 
    max_epochs, max_time or low_loss has been reached.  Return a list with 
    train and test loss, along with run time and the value of the bottleneck 
    neurons. Write the model to file. The fast option optimizes the gradient 
    descent for a faster search """

    # Define a generator for the minibatches than randomizes the data every
    # epoch. Don't yield the last batch if it has not the full batchsize
    def minibatches(data_train, batchsize):
        indices = np.arange(len(data_train))
        np.random.shuffle(indices)
        for start_idx in range(0, len(data_train) - batchsize + 1, batchsize):
            yield data_train[indices[start_idx:start_idx + batchsize]]

    # Define arrays over the layers of the theano variables.
    target = T.dmatrix()
    neurons = [T.dmatrix()]
    weights = []
    biases = []
    weights_grad = []
    biases_grad = []
    weights_momentum = []
    biases_momentum = []

    # Loop over each layer
    n_features = len(data_train[0])
    n_neurons = [n_features] + p.n_hidden_neurons + [n_features]
    bias_init = p.eps_init if p.has_bias else 0

    # Load pretrained parameters.
    if pretrained:
        print('Loading parameters from model {}...'.format(pretrained))
        with open('models/' + pretrained + '.pkl', 'rb') as f:
            network_pars = cPickle.load(f)[-1]
        weights_pre, biases_pre = network_pars

    # Initialise all the network parameters
    for i in range(len(n_neurons) - 1):
        if pretrained:
            weight = weights_pre[i]
            bias = biases_pre[i] 
        else:
            weight = init_weight(n_neurons[i], n_neurons[i+1], p.eps_init)
            bias = init_bias(n_neurons[i+1], bias_init)

        # Zero initial momentum
        weight_momentum = init_weight(n_neurons[i], n_neurons[i+1], 0)
        bias_momentum = init_bias(n_neurons[i+1], 0)

        # Append to lists
        weights.append(th.shared(weight))
        weights_momentum.append(th.shared(weight_momentum))
        biases.append(th.shared(bias))
        biases_momentum.append(th.shared(bias_momentum))

        # Define how to calculate the neurons in theano
        layer = p.activation_fn[i](neurons[i], weights[i], biases[i])
        neurons.append(layer)

    # Define the loss in theano
    loss = p.loss_fn(neurons[-1], target)

    # Define the gradients in theano
    for weight, bias in zip(weights, biases):
        weights_grad.append(T.grad(loss, weight))
        biases_grad.append(T.grad(loss, bias))

    # Define updates for weights parameters in theano
    updates_parameters = []
    updates_momentum = []
    for weight, weight_momentum, weight_grad in zip(weights, 
            weights_momentum, weights_grad):
        updates_parameters.append((weight, weight + weight_momentum))
        updates_momentum.append((weight_momentum, p.alpha*weight_momentum 
            - p.delta*weight_grad))

    # Define updates for biases parameters in theano
    if p.has_bias:
        for bias, bias_momentum, bias_grad in zip(biases, 
                biases_momentum, biases_grad):
            updates_parameters.append((bias, bias + bias_momentum))
            updates_momentum.append((bias_momentum, p.alpha*bias_momentum 
                - p.delta*bias_grad))

    # Define functions in theano
    update_parameters = th.function([], updates=updates_parameters)
    update_momentum = th.function([neurons[0], target], 
            updates=updates_momentum)
    give_loss = th.function([neurons[0], target], loss)
    give_neurons = th.function([neurons[0]], neurons)
    give_weights = th.function([], weights)
    give_biases = th.function([], biases)

    # Define parameters for the gradient descent
    loss_train = []
    loss_val = []
    used_data = []
    start_time = time.time()
    run_time = 0
    n_epochs = 0
    first_check = True
    converged = False
    optimal_rank = np.inf

    # Perform gradient descent over all data until converged or interrupted.
    print('Training the network...')
    try:
        while not converged:

            # Keep track of the epochs
            n_epochs += 1

            # Loop over minibatches
            for train_batch in minibatches(data_train, p.batchsize):

                # Do a gradient descent step with the mini-batch.
                update_momentum(train_batch,train_batch)
                update_parameters()

                # A more efficient code for grid_search
                if fast:

                    # Check convergence
                    if time.time() - start_time > p.max_time:
                        print('Converged in {} epochs!'.format(n_epochs))
                        return float(give_loss(data_val, data_val))

                    if time.time() - start_time > 150 and first_check:
                        first_check = False
                        lossx = float(give_loss(data_val, data_val))
                        if lossx > 0.075:
                            return lossx

                # Keep track of all losses and check all convergence criteria
                else:
                    # Store the data that is used in the current loss step
                    used_data.append(train_batch)

                    # If sufficient data has been used, update the loss lists.
                    if len(used_data) > p.loss_update_size//p.batchsize:

                        # Update loss_train and loss_val
                        used_data = np.concatenate(used_data)
                        train_loss = float(give_loss(used_data, used_data))
                        val_loss = float(give_loss(data_val, data_val))
                        loss_train.append(train_loss)
                        loss_val.append(val_loss)
                        used_data = []
                        print(('\rEpochs: {:3}, Run time: {:5.0f}s, '
                                'Train loss: {:1.6f}, Val loss: {:1.6f}')
                                .format(n_epochs, run_time, 
                                loss_train[-1], loss_val[-1]))

                        # Check recommender performance and save model if best
#                        m_rank = make_recipes(data_val, give_neurons, 'remove')
#                        if m_rank < optimal_rank:
#                            optimal_rank = m_rank
#                            print('Saving new recommender...')
#                            with open('models/recommender.pkl', 'wb') as f:
#                                proto = cPickle.HIGHEST_PROTOCOL
#                                cPickle.dump(give_neurons, f, protocol=proto)
#
                        # Checking all convergence criteria here
                        run_time = time.time() - start_time
                        if (
                            loss_val[-1] < p.low_loss or 
                            n_epochs > p.max_epochs or 
                            run_time > p.max_time
                        ):
                            print('\rGradient descent has converged!')
                            converged = True
                            break

    # Don't end program with KeyboardInterrupt, only end the training
    except KeyboardInterrupt:
        if fast:
            raise
        print('\rGradient descent ended by keyboard!')

    # Define a name for the model that has been trained
    model_dir = 'models/' + str(p)
    model_name = '{}/{:0.4f}'.format(p, loss_val[-1])
    print('Done training model: {}'.format(model_name))

    # Pickle all the important variables with cPickle
    network_parameters = give_weights(), give_biases()
    output = give_neurons, loss_train, loss_val, run_time, network_parameters
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs('figures/' + str(p))
    with open('models/' + model_name + '.pkl', 'wb') as f:
        cPickle.dump(output, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # Return model name
    return model_name
