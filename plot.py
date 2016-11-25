#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import matplotlib.pyplot as plt
import math
from numpy import genfromtxt
import matplotlib.patches as mpatches
from collections import Counter
from itertools import groupby
from cycler import cycler
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

from functions import power_law

#Not decided yet what to do with this code: Only show plot when asked or not
#plt.ioff()
plt.ion()


def fit_cost(cost_list):
    '''Fit a power_law to the cost_list to estimate the final cost 
    to which to network converges and the time it would take.'''

    fit_par = None
    y_values = None
    try:
        start = 1
        x_list = [i for i in range(len(cost_list))]
        fit_par, _ = curve_fit(power_law, x_list[start:], cost_list[start:])
        a, b, c, d = fit_par
        y_values = []
        for x_value in x_values:
            y_values.append(power_law(x_value, a, b, c, d))
        plt.plot(y_values, 'y')
        print(('Cost function fit: cost = {:.4f}*(steps - {:.4f})^'
               '(-{:.4f}) + {:.4f}').format(a, b, c, d))
    except:
        print('Around 100-150 points are needed to fit the cost funtion!')
        print('Try decreasing cost_update_size or train for longer.')

    return fit_par, y_values


def cost_function_plot(cost_train, cost_test, p, cost_zero=None,
        cost_svd=None, make_fit=True):
    '''Plot cost function, fit on it the cost of svd or predict-zero 
    and fit a power law through the points if asked.'''

    plot1, = plt.plot(cost_train, 'ro')
    plot2, = plt.plot(cost_test, 'go')
    plt.ylabel('Cost')
    plt.xlabel('Steps of {} observations'.format(p.cost_update_size))
    plt.title('Network:' + p.string)
    plt.legend([plot1, plot2], ['Train batches (averaged)', 'Test data'])
    text_position = (len(cost_train) - 1)*0.75
    if cost_zero:
        plt.axhline(y=cost_zero, xmin=0, xmax=1, hold=None)
        plt.annotate('Cost predict zero', (text_position, cost_zero))
    if cost_svd:
        plt.axhline(y=cost_svd, xmin=0, xmax=1, hold=None)
        plt.annotate('Cost predict svd', (text_position, cost_svd))
    fit_par = None
    if make_fit:
        fit_par, y_values = fit_cost(cost_test)
        if y_values:
            plt.plot(y_values, 'y')

    #Save image
    cost = cost_test[-1]
    plt.savefig('figures/costplot/{}_{:1.5f}.png'.format(p.string, cost))

    return fit_par


def biplot(data, regions, name, cost):
    '''Plot the reduced data set on a biplot or 3d plot, colored by origin'''

    colors = {'NorthAmerican':'lightgray',
            'LatinAmerican':'red',
            'SouthernEuropean':'purple',
            'WesternEuropean':'violet',
            'EasternEuropean':'blue',
            'NorthernEuropean':'turquoise',
            'African':'black',
            'MiddleEastern':'green',
            'SouthAsian':'greenyellow',
            'SoutheastAsian':'yellow',
            'EastAsian':'orange'}

    #Create a list with the unique regions sorted in descending frequency
    region_counts = Counter(regions)
    regions_list = sorted(region_counts, key=region_counts.get, reverse=True)

    #Create a list containing the lists of recipes per region
    recipes_by_region = [[] for _ in range(len(regions_list))]
    for recipe, region in zip(data, regions):
        recipes_by_region[regions_list.index(region)].append(recipe)

    #Determine dimensions of data and make a corresponding type of plot
    dim = len(data[0])
    if not dim in [2, 3]:
        return
    if dim == 2: #2D plot
        fig, ax = plt.subplots()
    elif dim == 3: #3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    #Plot the recipies for each group, highest frequency first
    for i, recipe in enumerate(recipes_by_region):
        ax.plot(*zip(*recipe), marker='.', linestyle='', 
                color=colors[regions_list[i]], ms=4, label=regions_list[i])
    ax.legend(loc=4, markerscale=3, prop={'size':6})

    #Save image
    plt.savefig('figures/biplot/{}_{:1.5f}.png'.format(name, cost))
