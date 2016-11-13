#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import numpy as np
from numpy import random as r
import matplotlib.pyplot as plt
import math
plt.ion()
from numpy import genfromtxt
import pandas
import matplotlib.patches as mpatches
import statistics as st
from collections import Counter
from itertools import groupby
from cycler import cycler

#Only show plot when asked
#plt.ioff()

def cost_function_plot(cost_train, cost_test, costfn, cost_update_size,
        cost_zero=None, cost_svd=None):
    '''Plot cost function'''

    plot1, = plt.plot(cost_train, 'ro')
    plot2, = plt.plot(cost_test, 'go')
    plt.ylabel('Cost')
    plt.xlabel('Steps of {} observations'.format(cost_update_size))
    plt.title('Plot of ' + costfn.__name__ + ' cost function')
    plt.legend([plot1, plot2], ['Average train batches', 'Test data'])
    text_position = (len(cost_train) - 1)*0.75
    if cost_zero:
        plt.axhline(y=cost_zero, xmin=0, xmax=1, hold=None)
        plt.annotate('Cost predict zero', (text_position, cost_zero))
    if cost_svd:
        plt.axhline(y=cost_svd, xmin=0, xmax=1, hold=None)
        plt.annotate('Cost predict svd', (text_position, cost_svd))

def biplot(data, regions):
    '''Plot the bottleneck hidden layer'''

    #Make figure and axis
    fig, ax = plt.subplots()

    #Set colors of the regions
    ax.set_prop_cycle(cycler('color', ['lightgray', 'purple', 'red',
        'violet', 'orange', 'green', 'greenyellow', 'yellow', 'blue',
        'black', 'turquoise']))

    #Create a list with the unique regions sorted in descending frequency
    region_counts = Counter(regions)
    unique_regions = sorted(region_counts, key=region_counts.get, 
            reverse=True)

    #Create a list containing the lists of recipes per region
    recipes_by_region = [[] for _ in range(len(unique_regions))]
    for recipe, region in zip(data, regions):
        recipes_by_region[unique_regions.index(region)].append(recipe)

    #Plot the recipies for each group, highest frequency first
    for i, recipe in enumerate(recipes_by_region):
        x, y = zip(*recipe)
        ax.plot(x, y, marker='.', linestyle='', 
                ms=4, label=unique_regions[i])
    ax.legend(loc=4, markerscale=3, prop={'size':6})
