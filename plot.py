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

from functions import power_law

#Not sure yet what to do with this code: Only show plot when asked
#plt.ioff()
plt.ion()

def fit(cost_list):
    '''Fit a power_law to the cost_list to estimate the final cost 
    to which to network converges and the time it would take.'''

    #x_list = [i for i in range(len(cost_list))]



def cost_function_plot(cost_train, cost_test, p, cost_zero=None,
        cost_svd=None, make_fit=True):
    '''Plot cost function, fit on it the cost of svd or predict-zero 
    and fit a power law through the points if asked.'''

    plot1, = plt.plot(cost_train, 'ro')
    plot2, = plt.plot(cost_test, 'go')
    plt.ylabel('Cost')
    plt.xlabel('Steps of {} observations'.format(p.cost_update_size))
    plt.title('Network:' + p.string)
    plt.legend([plot1, plot2], ['Average train batches', 'Test data'])
    text_position = (len(cost_train) - 1)*0.75
    if cost_zero:
        plt.axhline(y=cost_zero, xmin=0, xmax=1, hold=None)
        plt.annotate('Cost predict zero', (text_position, cost_zero))
    if cost_svd:
        plt.axhline(y=cost_svd, xmin=0, xmax=1, hold=None)
        plt.annotate('Cost predict svd', (text_position, cost_svd))

    #Fit power_law through costfunction
    par_fit = None
    if make_fit:
        try:
            x_values = [i for i in range(len(cost_test))]
            #fit on all but 5 first values. More reliable.
            par_fit, _ = curve_fit(power_law, x_values[5:], cost_test[5:])
            a, b, c, d = par_fit
            y_values = []
            for x_value in x_values:
                y_values.append(power_law(x_value, a, b, c, d))
            plt.plot(y_values, 'y')
            print(('Cost function fit: cost = {:.2f}*(steps - {:.2f})^'
                   '(-{:.2f}) + {:.2f}').format(a, b, c, d))
        except:
            print('Around 100 points needed to fit cost funtion!')
            print('Try decreasing cost_update_size or train for longer.')

    #Save image
    plt.savefig('figures/costplot/{}_{:1.5f}.png'.format(p.string, 
        cost_test[-1]))
    print(par_fit)
    return par_fit


def biplot(data, regions, name, cost):
    '''Plot the reduced data set on a biplot, colored by origin'''

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

    #Save image
    plt.savefig('figures/biplot/{}_{:1.5f}.png'.format(name, cost))
