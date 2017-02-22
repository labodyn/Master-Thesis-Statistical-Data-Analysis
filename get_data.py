#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import pandas as pd
import numpy as np

def get_data(remove_northamerican=False, standardize=False):
    """ Import the data, return as numpy array. The ingredients names are 
    stored in 'ingredients'.  'Data' contains the ingredients for each recipe,
    coded as 0/1. 'Regions' contains the region of origin of the recipe. """

    print('Reading in datafile...', end = ' ')

    # Read in data
    df = pd.read_csv('data/ReceptenBinair_minimun2ingredients.csv', sep=';')
    values = df.values
    ingredients = np.array(df.columns)[1:]

    # Shuffle the data with seed
    np.random.seed(42)
    np.random.shuffle(values)

    # Split into the 0/1 coded ingredients matrix and the regions
    data = values[:,1:].astype(int)
    regions = values[:,0]

    # Standardize
    if standardize:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Remove NorthAmerican recipes
    if remove_northamerican:
        data = data[regions != 'NorthAmerican']
        regions = regions[regions != 'NorthAmerican']

    # Print out dimensions and return regions, data and ingredients
    print('Got {} recipies and {} ingredients.'.format(
        len(data), len(ingredients)))
    return data, regions, ingredients
