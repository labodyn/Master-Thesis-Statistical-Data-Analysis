#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import pandas as pd
import numpy as np
from collections import OrderedDict

from myclasses import MyBunch
from train import predict_zero, perform_svd, train_autoencoder, performance_LDA
from plot import cost_function_plot, biplot
from functions import rectifier, sigmoid, linear, cross_entropy, least_squares


def get_parameters():
    '''Returns all the parameters of the network and gradient descent.'''

    p = OrderedDict()

    #Network settings
    p['has_bias'] = True #Add bias to the layers or not
    p['cost_fn'] = cross_entropy
    p['activation_fn'] = [rectifier, rectifier, linear, rectifier, rectifier, sigmoid]
    p['n_hidden_neurons'] = [100, 100, 2, 100, 100]

    #Gradient descent parameters
    p['test_size'] = 0.3
    p['eps_init'] = 0.1 #Half of the init range of the network param
    p['batch_size'] = 10 #8 #Number of observations to use for gradient descent
    p['alpha'] = 0.8 #Inertie coefficient.
    p['delta'] = 2 #Learning rate
    p['max_loops'] = 800 #Maximum number of loops over all the data
    p['max_time'] = 18000 #Maximum time
    p['low_cost'] = 0.04 #Lowest cost
    p['cost_update_size'] = 2000 #Number of data to use before updating cost
    p['plot_svd'] = True #Plot the cost of data reduction with an SVD

    return MyBunch(p)


def search_grid():
    '''Search a grid of the network and graddesc parameters 
    to find the best ones.'''

    pass

    #Define grid
    batchsize_grid = []

    #List to store the convergence times
    times = []


def write_out_results(data_reduced, region):
    '''Write out the reduced dataset to a file.'''

    pass
    ''' print("Writing out hidden neuron values")
    outputfile = open('output/neuronvalues.txt', 'w')
    outputfile.write('neuron1\tneuron2\tgroup\n')
    for i in range(len(groupoutput)):
        outputfile.write((str(data_reduced[i][0])+'\t'
            str(hidlayoutput[i][1])+'\t'))
        outputfile.write(groupoutput[i]+'\n')

    outputfile.close() '''


def process_data():
    '''Import the data set and perform different techniques on it like 
    auto-encoders and singular value decomposition.'''

    #Get parameters
    p = get_parameters()
    p.print_out()
    p.make_network_string()

    #Import data
    df = pd.read_csv('datasets/ReceptenBinair_minimun2ingredients.csv',\
            sep=';', index_col=0)
    data = df.values
    regions = df.index
    ingredient_names = df.columns
    print('Read in datafile with {} recipies and {} ingredients.'
            .format(len(data), len(ingredient_names)))

    #Train the network
    results = train_autoencoder(data, p)
    bottleneck_neurons, cost_train, cost_test, running_time = results


    #Try out some linear reduction methods to compare with
    #cost_zero = predict_zero(data, p)
    #cost_svd, data_svd_reduced = perform_svd(data, p)

    #Visualize results
    fit = cost_function_plot(cost_train, cost_test, p)#, 
        #cost_zero=cost_zero, cost_svd=cost_svd) 
    biplot(bottleneck_neurons, regions, p.string, cost_test[-1])
    #biplot(data_svd_reduced, regions, 'svd', cost_svd)


if __name__ == '__main__':
    process_data()
