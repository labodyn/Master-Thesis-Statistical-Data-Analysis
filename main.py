#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import pandas as pd
import numpy as np
import pickle
import math
from random import randint, choice, uniform
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
from six.moves import cPickle
from sklearn.preprocessing import normalize

# Import from own modules
from get_data import get_data
from myclasses import NNParameters
from train_autoencoder import train_autoencoder
from train_other import train_zero_prediction, train_svd, train_qda, train_knn
from plot import loss_plot, biplot, roc_plot
from functions import rectifier, sigmoid, linear, cross_entropy, least_squares
from performance import predict_regions, make_recipes

def get_parameters():
    """ Returns all the parameters of the network and gradient descent 
    as an object of the class NNParameters. """

    # Network settings
    has_bias = True # Add bias to the layers or not
    loss_fn = cross_entropy
    activation_fn = [rectifier, linear, rectifier, sigmoid]
    n_hidden_neurons = [100, 2, 100]

    # Gradient descent parameters
    eps_init = 0.288  # Half of the init range of the network param
    batchsize = 12  # Number of observations to use for gradient descent
    alpha = 0.924  # Inertie coefficient.
    delta = 1.5  # Learning rate
    max_epochs = 800  # Maximum number of loops over all the data
    max_time = 5000  # Maximum time
    low_loss = 0.0052  # Lowest loss
    loss_update_size = 10000 # Number of obs to use before updating loss

    return NNParameters(locals())

def adaptive_random_search(max_time=99, pretrained=None, return_par_opt=False):
    """ Adaptive random search to optimize the hyperparameters. """

    # Get parameters
    p = get_parameters()
    p.max_time = max_time

    # Define the upper bound of symmetric ranges over which to vary.
    # These should start wide and narrow down when close to a good solution
    bs_r = 3  # Additive
    al_r = 1.2  # Multiplicative, using 1/(1 - alpha)
    de_r = 1.3  # Multiplicative
    ei_r = 1.15  # Multiplicative

    # Function to return the best set of parameters, given a list with 
    # parameter with the hyperparameters and their convergence loss
    def get_par_opt(par_list):
        return sorted(par_list)[0][1:-1]

    # Load the list of hyperparameter searches, make new list if none exists
    file_name = 'hyper_parameter_search/' + str(p) + '.pkl'
    try:
        with open(file_name, 'rb') as f:
            par_list = pickle.load(f)
            par_opt = get_par_opt(par_list)
    except FileNotFoundError:
        print('No parameter search list exists. Making new one...')
        par_list = []
        par_opt = p.batchsize, p.alpha, p.delta, p.eps_init

    if return_par_opt:
        return par_opt

    # Get the data
    data, regions, ingredients = get_data()

    # Separate 2500 test and 2500 val recipes
    data_val, data_train = data[2500:5000], data[5000:]

    # Define the string to print out
    out_str = ('\rLoss: {:0.5f}, Batch: {:2}, Alpha: {:1.3f}, '
            'Delta: {:2.3f}, Eps: {:1.3f}, Time: {:4}')

    # Drawn new parameters in the region of the optimal parameters so far
    def draw_parameters(par_opt):
        batchsize_opt, alpha_opt, delta_opt, eps_init_opt = par_opt
        batchsize = max(batchsize_opt + randint(-bs_r, bs_r), 1)
        alpha = max(1 - (1-alpha_opt)*uniform(1/al_r, al_r), 0)
        delta = delta_opt*uniform(1/de_r, de_r)
        eps_init = eps_init_opt*uniform(1/ei_r, ei_r)
        return batchsize, alpha, delta, eps_init

    print('Starting hyperparameter search until stopped by keyboard...')
    try:
        while True:

            # Draw new parameters 
            par = draw_parameters(par_opt)

            # Run autoencoder with the parameters
            p.batchsize, p.alpha, p.delta, p.eps_init = par
            loss = train_autoencoder(data_train, data_val, p, pretrained, True)
            loss = 1 if math.isnan(loss) else loss

            # Add them to the list and save the list
            all_par = loss, *par, max_time
            print(out_str.format(*all_par))
            par_list.append(all_par)
            with open(file_name, 'wb') as f:
                pickle.dump(par_list, f)

            # Find optimal parameters so far
            par_opt = get_par_opt(par_list)

    except KeyboardInterrupt:
        print('\rAdaptive random search interrupted by keyboard!')

    # Print out all combination sorted on loss
    print('Combinations for network {}:'.format(p))
    for all_par in sorted(par_list):
        print(out_str.format(*all_par))

def main(model_name=None, optimal_par=False, pretrained=None):
    """ Import the data set and perform different unsupervised machine learning
    techniques on it (autoencoder and singular value decomposition) to reduce
    the data set to a small number of continuous features. Use a simple 
    supervised classifier (LDA, KNN, ...) to predict the region of the recipes
    on the raw dataset and both reduced datasets and measure the performance 
    of this. Use the autoencoder network to reconstruct an ingredient that was 
    removed from a recipe and measure performance. """

    # Get the parameters defining the network and gradient descent
    p = get_parameters()

    # Get optimal gradient descent parameters from adaptive_random_search
    if optimal_par:
        par_opt = adaptive_random_search(return_par_opt=True)
        p.batchsize, p.alpha, p.delta, p.eps_init = par_opt

    p.print_out()

    # Get the data
    data, regions, ingredients = get_data()

    # Separate 1000 test and 1000 val recipes
    data_test = data[:2500]
    data_val = data[2500:5000]
    data_train = data[5000:]

    # --------------- Train autoencoder ---------------
    if model_name is None:
        # Train the autoencoder
        model_name  = train_autoencoder(data_train, data_val, p, pretrained)

    # Load the model
    with open('models/' + model_name + '.pkl', 'rb') as f:
        give_neurons, loss_train, loss_val, run_time = cPickle.load(f)[:-1]
        #give_neurons = cPickle.load(f)
    data_ae = give_neurons(data.tolist())[p.bottleneck_index]

    # --------------- Train a zero predictor and SVD ---------------
    loss_zero = train_zero_prediction(data_val, p)
    loss_svd = 0.0161952433812
    #loss_svd, V_k = train_svd(data_train, data_val, p)
    #data_svd = np.dot(data, V_k)

    # -------------- Do stuff with all the models ----------------

    # Make ROC plot of reconstruction
    #data_val_ae = give_neurons(data_val)[-1]
    #roc_plot(data_val, data_val_ae, model_name)

    # Make plot of the loss function for the training of the autoencoder
    #loss_plot(loss_train, loss_val, p, model_name, loss_zero, loss_svd)

    # Do region prediction on the raw data set and the recuded data sets.
    #biplot(data_ae, regions, model_name)
    #biplot(data_svd, regions, 'svd')
    predict_regions(give_neurons, p, model_name)
    #predict_regions(V_k, p, 'svd')
    #predict_region(raw)
    #predict_region(svd)

    #print('Reconstructing recipes with a missing/extra ingredient...')
    #make_recipes(data_val, give_neurons, 'remove', ingredients)
    #make_recipes(data_test, give_neurons, 'remove', ingredients, show=40)
    #make_recipes(data_val, give_neurons, 'add', ingredients)
    #make_recipes(data_test, give_neurons, 'add', ingredients, show=20)
    #make_recipes(svd)

    # Prevent plt from closing the images when program comes to its end
    input('Press any key to close images and end program')

if __name__ == '__main__':

    # To train a network with a lot of sigmoid functions, train a network with
    # a lot rectifier instead and use that as pretrained network.

    #adaptive_random_search(max_time=800, pretrained='bces100s50l2s50s100s/0.0519')
    main(model_name='bces100s50l2s50s100s/0.0519')
