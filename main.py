#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import pandas as pd
import numpy as np
import pickle
import os
import math
from random import choice 
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
from six.moves import cPickle
from sklearn.preprocessing import normalize

from myclasses import NNParameters
from train_model import perform_zero_prediction, perform_svd, perform_autoencoding, performance_lda, performance_knn
from plot import cost_function_plot, biplot, roc_plot
from functions import rectifier, sigmoid, linear, cross_entropy, least_squares

def get_parameters():
    """ Returns all the parameters of the network and gradient descent 
    as an object of the class NNParameters. """

    p = OrderedDict()

    # Network settings
    p['has_bias'] = True # Add bias to the layers or not
    p['cost_fn'] = cross_entropy
    p['activation_fn'] = [rectifier, rectifier, linear, rectifier, rectifier,
            sigmoid]
    p['n_hidden_neurons'] = [400, 200, 2, 200, 400]

    # Gradient descent parameters
    p['test_size'] = 0.002  # Fraction to test the cost
    p['eps_init'] = 0.025  # Half of the init range of the network param
    p['batch_size'] = 25  # Number of observations to use for gradient descent
    p['alpha'] = 0.9  # Inertie coefficient.
    p['delta'] = 0.15 #1  # Learning rate
    p['max_loops'] = 800  # Maximum number of loops over all the data
    p['max_time'] = 3000  # Maximum time
    p['low_cost'] = 0.0052  # Lowest cost
    p['cost_update_size'] = 20000  # Number of data to use before updating cost

    return NNParameters(p)

def get_data(remove_northamerican=False, standardize=False):
    """ Import the data, return as numpy array. The ingredients names are 
    stored in 'ingredients'.  'Data' contains the ingredients for each recipe,
    coded as 0/1. 'Regions' contains the region of origin of the recipe. """

    assert not (remove_northamerican and standardize), 'code does not work for'

    # Read in data
    df = pd.read_csv('data/ReceptenBinair_minimun2ingredients.csv', sep=';')
    values = df.values
    ingredients = np.array(df.columns)[1:]

    # Remove NorthAmerican recipes
    if remove_northamerican:
        values = values[values[:,0] != 'NorthAmerican']

    # Shuffle the data with seed
    np.random.seed(42)
    np.random.shuffle(values)

    # Split into the 0/1 coded ingredients matrix and the regions
    data = values[:,1:].astype(int)
    regions = values[:,0]

    # Standardize
    if standardize:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Print out dimensions and return regions, data and ingredients
    out_text = 'Read in datafile with {} recipies and {} ingredients.'
    print(out_text.format(len(data), len(ingredients)))
    return data, regions, ingredients

def write_out_results(data_reduced, regions):
    """ Write out the reduced dataset to a file. """

    print('Writing out hidden neuron values...')
    output_file = open('output/neuronvalues.txt', 'w')

    # Make header
    output_file.write('Regions\n')
    for neuron_nr in range(1, len(data_reduced[0]) + 1):
        output_file.write('Neuron_value_{}\t'.format(neuron_nr))

    # Write out values
    for features, region in zip(data_reduced, regions):
        output_file.write('{}\n'.format(region))
        for feature in features:
            output_file.write('{}\t'.format(feature))
    output_file.close() 

def make_recipe(model, ingredients, n_iter=1000, cutoff=0.1, *args):
    """ function description comes here """

    # Give ingredients list if no input is given.
    if not args:
        print('No ingredients given. Ingredients list:')
        print(', '.join(ingredients))
        return

    # Convert the given ingredient names to a 0/1-coded list
    recipe = [0]*len(ingredients)
    for ingredient in args:
        ingredient_lower = ingredient.lower()
        if ingredient_lower in ingredients:
            recipe[ingredients.index(ingredient_lower)] = 1
        else:
            print('Ingredient not found:', ingredient_lower)

    # Make new recipe by putting the recipe several times through the network.
    recipe = [recipe] #Model requires the input variables as a list
    for _ in range(n_iter):
        out_layer = model(recipe)[-1]
        recipe = out_layer.tolist()
    recipe = recipe[0]

    # Get the ingredient names and sort on reconstruction %
    final_list = []
    for i, ingredient_pct in enumerate(recipe[0]):
        if ingredient_pct > cutoff:
            final_list.append((ingredient_pct, ingredients[i]))
    final_list.sort(reverse = True)

    # Print out ingredients of the final recipe
    print('Suggested ingredients for recipe:')
    for element in final_list:
        print('{:20} ({:.1f}%)'.format(element[1], element[0]*100))

def random_grid_search():
    """ A random version of grid search to optimize the hyperparameters. 
    A grid is defined for each parameter, after which all different
    combinations of parameters are tested. The parameter combinations will be
    picked in a random order without repeating certain combinations. If the 
    grid search is aborted with KeyboardInterrupt, the random component will 
    make it much more likely a good combination is found within the timeframe.
    """

    # Get data and parameters
    data, regions, ingredients = get_data()
    p = get_parameters()

    # set test_size. Too small will give inaccurate results
    p.test_size = 0.1

    # Keep track of tried combinations of parameters
    file_name = 'grid_search/' + str(p) + '.pkl'
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            gs_par_list, gs_cost_list = pickle.load(f)
    else:
        gs_par_list = []
        gs_cost_list = []

    # Define grids
    batch_size_grid = [10, 15, 20, 25, 30, 50, 75]
    alpha_grid = [0.8, 0.85, 0.9, 0.925, 0.95, 0.975]
    delta_grid = [0.01, 0.025, 0.05, 0.075, 0.1]
    eps_init_grid = [0.01, 0.025, 0.05, 0.1]
    max_time_grid = [1500]  # ...

    out_str = ('Batch: {:2}, Alpha: {:1.2f}, Delta: {:2.3f}, '
            'Eps: {:1.3f}, Time: {:4}, Cost: {:0.5f}')

    def draw_parameters():
        batch_size = choice(batch_size_grid)
        alpha = choice(alpha_grid)
        delta = choice(delta_grid)
        eps_init = choice(eps_init_grid)
        max_time = choice(max_time_grid)
        return batch_size, alpha, delta, eps_init, max_time

    try:
        while True:

            # Draw until new parameter set
            par = draw_parameters()
            while par in gs_par_list:
                par = draw_parameters()

            # Run autoencoder with the parameters
            p.batch_size, p.alpha, p.delta, p.eps_init, p.max_time = par
            cost = perform_autoencoding(data, p, grid_search=True)
            if math.isnan(cost):
                cost = 1

            # Add them to the list and save the list
            print(out_str.format(*par, cost))
            gs_par_list.append(par)
            gs_cost_list.append(cost)
            with open(file_name, 'wb') as f:
                pickle.dump((gs_par_list, gs_cost_list), f)

    except KeyboardInterrupt:
        print('Random grid search interrupted by keyboard!')

    # Print out all combination
    for cost, par in sorted(zip(gs_cost_list, gs_par_list)):
        print(out_str.format(*par, cost))

def knn_on_raw():
    """ KNN on the raw ingredients data to predict the origin. KNN scales 
    incredibly bad to higher dimensions, so this can take long.  """

    # Get data without North American recipes 
    data, regions, ingredients = get_data(remove_northamerican=True)

    data_train, data_test = train_test_split(data, test_size=0.1, 
            random_state=42)

    regions_train, regions_test = train_test_split(regions, 
            test_size=0.1, random_state=42)

    performance_knn(data_train, data_test, regions_train, regions_test,
    'KNN on the raw data')

def predict_region(give_neurons, regions):
    """ Predict the regions of the recipes using the neurons of the bottleneck
    layer in the autoencoder. Filter out NorthAmerican for region prediction. 
    NorthAmerican contains recipes of all regions and dominates in numbers, 
    which makes the prediction trivial."""

    # Make biplot of the bottleneck neurons and SVD reduction
    #biplot(bottleneck_neurons, regions, model_name)
    #biplot(data_svd, regions, 'svd_{:0.4f}'.format(cost_svd))
    # Get the bottlneck neurons of all data

    all_neurons = give_neurons(data.tolist())
    bottleneck_neurons = all_neurons[p.bottleneck_index]

    filtered = []
    regions_filtered  = []
    for features, region in zip(bottleneck_neurons, regions):
        if region != 'NorthAmerican':
            features_filtered.append(features)
            regions_filtered.append(region)

    #biplot(features_filtered, regions_filtered, model_name + '_filter')

    # Measure performance without NorthAmerican. Different train/test split 
    # than autoencoder training since the outcome variable 'region' wasn't used
    data_train, data_test = train_test_split(features_filtered, test_size=0.1,
            random_state=42)
    regions_train, regions_test = train_test_split(regions_filtered, 
            test_size=0.1, random_state=42)

    # LDA
    #regions_predicted = performance_lda(data_train, data_test, regions_train, 
    #        regions_test)
    #biplot(data_test, regions_predicted, 'lda_' + model_name)

    # KNN
    regions_predicted = performance_knn(data_train, data_test, regions_train, 
            regions_test, model_name)
    #biplot(data_test, regions_predicted, 'knn_' + model_name)

    #input('Press any key to close figures')



def main(model_name=None, optimal_par=False):
    """ Import the data set and perform different unsupervised machine learning
    techniques on it (auto-encoders and singular value decomposition) to reduce
    the data set to a small number of continuous features. Use a simple 
    supervised classifier (LDA, KNN, ...) to predict the region of the recipes
    on the raw data set and the reduced datasets and measure performance. Use 
    the reduced datasets to reconstruct an ingredient that was removed from a
    recipe and measure performance. """

    # Get parameters
    p = get_parameters()
    if optimal_par:
        pass  # Get optimal gradient descent parameters from random_grid_search
    p.print_out()

    # Get the data.
    data, regions, ingredients = get_data()

    # Separate 1000 recipes for the recipe reconstruction phase
    data_reconstruct, data_autoencoder = data[:1000], data[1000:]
    _, regions_reconstruct = regions[:1000], regions[:1000]

    # Train a new autoencoder network if no network model is given and load 
    # the results of the autoencoder model, saved as pickle file
    if model_name is None:
        model_name  = perform_autoencoding(data_autoencoder, p) 
    with open('models/' + model_name + '.pkl', 'rb') as f:
        output = cPickle.load(f)
    give_neurons, cost_train, cost_test, y_test, y_pred, run_time = output

    # Make ROC plot
    #roc_plot(y_test, y_pred, model_name)

    # Try out some linear reduction methods to compare with
    #cost_zero = perform_zero_prediction(data, p)
    #cost_svd, data_svd = perform_svd(data, p)

    # Make plot of the loss function of the training of the autoencoder
    cost_function_plot(cost_train, cost_test, p, model_name)
    #    cost_zero=cost_zero, cost_svd=cost_svd)

    # Do region prediction on the raw data set and the recuded data sets.
    #predict_region(give_neurons, regions):
    #predict_region(data_svd, regions):

if __name__ == '__main__':

    #random_grid_search()
    main()
    #knn_on_raw()
