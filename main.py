#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
from six.moves import cPickle

from myclasses import NNParameters
from train_model import perform_zero_prediction, perform_svd, perform_autoencoding, performance_lda
from plot import cost_function_plot, biplot
from functions import rectifier, sigmoid, linear, cross_entropy, least_squares

def get_parameters():
    '''Returns all the parameters of the network and gradient descent.'''

    p = OrderedDict()

    #Network settings
    p['has_bias'] = True #True #Add bias to the layers or not
    p['cost_fn'] = cross_entropy
    p['activation_fn'] = [rectifier, rectifier, linear, rectifier, rectifier, sigmoid]
    p['n_hidden_neurons'] = [200, 50, 25, 50, 200]

    #Gradient descent parameters
    p['test_size'] = 0.02 #Fraction to test the cost
    p['eps_init'] = 0.05 #Half of the init range of the network param
    p['batch_size'] = 10 #8 #Number of observations to use for gradient descent
    p['alpha'] = 0.8 #Inertie coefficient.
    p['delta'] = 0.5 #1 #2 #Learning rate
    p['max_loops'] = 800 #Maximum number of loops over all the data
    p['max_time'] = 18000 #Maximum time
    p['low_cost'] = 0.0 #Lowest cost
    p['cost_update_size'] = 50000 #Number of data to use before updating cost

    return NNParameters(p)


def search_grid():
    '''Search a grid of the network and graddesc parameters 
    to find the best ones.'''

    pass

    #Define grid
    batchsize_grid = []

    #List to store the convergence times
    times = []


def write_out_results(data_reduced, regions):
    '''Write out the reduced dataset to a file.'''

    print('Writing out hidden neuron values...')
    output_file = open('output/neuronvalues.txt', 'w')

    #Make header
    for neuron_nr in range(1, len(data_reduced[0]) + 1):
        output_file.write('Neuron_value_{}\t'.format(neuron_nr))
    output_file.write('Regions\n')

    #Write out values
    for features, region in zip(data_reduced, regions):
        for feature in features:
            output_file.write('{}\t'.format(feature))
        output_file.write('{}\n'.format(region))
    output_file.close() 

def make_recipe(*args, cutoff=0.1):
    ''''''

    #Load ingredients list
    df = pd.read_csv('datasets/ReceptenBinair_minimun2ingredients.csv',\
            sep=';', index_col=0)
    ingredients = df.columns.tolist()

    #give ingredients list if no input is given.
    if not args:
        print('No ingredients given. Ingredients list:')
        print(', '.join(ingredients))
        return

    #Load network model
    f = open('output/model.pkl', 'rb')
    put_through_network = cPickle.load(f)
    f.close()

    #Convert ingredient to list
    recipe = [0]*len(ingredients)
    for ingredient in args:
        if ingredient.lower() in ingredients:
            recipe[ingredients.index(ingredient.lower())] = 1
        else:
            print('Ingredient not found:', ingredient)

    #Make recipe
    recipe = [recipe]
    for _ in range(1000):
        recipe = put_through_network(recipe).tolist()

    #Print out ingredients of recipe
    final_list = []
    print('Suggested ingredients:')
    for i, ingredient_pct in enumerate(recipe[0]):
        if ingredient_pct > cutoff:
            final_list.append((ingredient_pct, ingredients[i]))
    final_list.sort(reverse = True)
    for element in final_list:
        print('{:20} ({:.1f}%)'.format(element[1], element[0]*100))

def process_data():
    '''Import the data set and perform different unsupervised machine learning
    techniques on it (auto-encoders and singular value decomposition).
    Measure performance with a simple supervised classifier (LDA, KNN, ...). 
    Use autoencoder network to create new recipies.'''

    #Get parameters
    p = get_parameters()
    p.print_out()

    #Import data
    df = pd.read_csv('datasets/ReceptenBinair_minimun2ingredients.csv',\
            sep=';', index_col=0)
    data = df.values
    regions = df.index
    unique_regions = sorted(set(regions))
    ingredients = df.columns
    print('Read in datafile with {} recipies and {} ingredients.'
            .format(len(data), len(ingredients)))

    #Train the autoencoder network
    results = perform_autoencoding(data, p)
    bottleneck_neurons, cost_train, cost_test, running_time = results

    #Try out some linear reduction methods to compare with
    #cost_zero = perform_zero_prediction(data, p)
    #cost_svd, data_svd_reduced = perform_svd(data, p)

    #Visualize results
    fit = cost_function_plot(cost_train, cost_test, p)#, 
        #cost_zero=cost_zero, cost_svd=cost_svd) 
    biplot(bottleneck_neurons, regions, p.string, cost_test[-1])
    #biplot(data_svd_reduced, regions, 'svd', cost_svd)

    #Write out results
    write_out_results(bottleneck_neurons, regions)

    #Filter out NorthAmerican.
    features_filtered = []
    regions_filtered  = []
    for features, region in zip(bottleneck_neurons, regions):
        if region != 'NorthAmerican':
            features_filtered.append(features)
            regions_filtered.append(region)

    #Measure Performance without NorthAmerican. Different train/test split.
    data_train, data_test = train_test_split(features_filtered, 
            test_size=0.3, random_state=16)
    regions_train, regions_test = train_test_split(regions_filtered, 
            test_size=0.3, random_state=16) 

    #lda
    regions_predicted = performance_lda(data_train, data_test, regions_train, 
            regions_test)
    biplot(data_test, regions_predicted, p.string + '_lda', cost_test[-1])

    input('Press any key to continue')

if __name__ == '__main__':
    process_data()
