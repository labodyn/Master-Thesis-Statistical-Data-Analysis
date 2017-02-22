#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import genfromtxt
import matplotlib.patches as mpatches
from collections import Counter
from itertools import groupby
from cycler import cycler
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve

from functions import power_law

#Not decided yet what to do with this code: Only show plot when asked or not
#plt.ioff()
plt.ion()


def fit_loss(loss_list):
    """ EXPERIMENTAL, OUT OF USE ATM...  Fit a power_law to the loss_list 
    to estimate the final loss to which to network converges and the time 
    it would take."""

    fit_par = None
    y_values = None
    try:
        start = 1
        x_list = [i for i in range(len(loss_list))]
        fit_par, _ = curve_fit(power_law, x_list[start:], loss_list[start:])
        a, b, c, d = fit_par
        y_values = []
        for x_value in x_values:
            y_values.append(power_law(x_value, a, b, c, d))
        plt.plot(y_values, 'y')
        print(('Cost function fit: loss = {:.4f}*(steps - {:.4f})^'
               '(-{:.4f}) + {:.4f}').format(a, b, c, d))
    except: #convergence error, which type?
        print('Around 100-150 points are needed to fit the loss funtion!')
        print('Try decreasing loss_update_size or train for longer.')

    return fit_par, y_values

def loss_plot(loss_train, loss_val, p, model_name, loss_zero=None,
        loss_svd=None):
    """ Plot loss function and fit on it the loss of svd and predict-zero """

    plt.figure()
    plot1, = plt.plot(loss_train, 'ro')
    plot2, = plt.plot(loss_val, 'go')
    plt.ylabel(p.loss_fn.__name__.title() + ' loss')
    plt.xlabel('Steps of {} observations'.format(p.loss_update_size))
    plt.legend([plot1, plot2], ['Train batches (averaged)', 'Test data'])
    text_position = (len(loss_train) - 1)*0.75
    if loss_zero:
        plt.axhline(y=loss_zero, xmin=0, xmax=1, hold=None)
        plt.annotate('Loss predict zero', (text_position/3.1, loss_zero))
    if loss_svd:
        plt.axhline(y=loss_svd, xmin=0, xmax=1, hold=None)
        plt.annotate('Loss predict svd', (text_position/3.1, loss_svd))
    plt.savefig('figures/' + model_name + '_loss.png')

def biplot(data, regions, model_name):
    """ Plot the reduced data set on a biplot or 3d plot, colored by origin"""

    # Define the colors for each region
    colors = {
            'NorthAmerican':'lightgray',
            'LatinAmerican':'red',
            'SouthernEuropean':'purple',
            'WesternEuropean':'violet',
            'EasternEuropean':'blue',
            'NorthernEuropean':'turquoise',
            'African':'black',
            'MiddleEastern':'green',
            'SouthAsian':'greenyellow',
            'SoutheastAsian':'yellow',
            'EastAsian':'orange'
    }

    # Create a list with the unique regions sorted in descending frequency
    region_counts = Counter(regions)
    regions_list = sorted(region_counts, key=region_counts.get, reverse=True)

    # Create a list containing the lists of recipes per region
    recipes_by_region = [[] for _ in range(len(regions_list))]
    for recipe, region in zip(data, regions):
        recipes_by_region[regions_list.index(region)].append(recipe)

    # Determine dimensions of data and make a corresponding type of plot
    dim = len(data[0])
    if not dim in (2, 3):
        print('Wrong dimensions for biplot!')
        return
    if dim == 2:  # 2D plot
        fig, ax = plt.subplots()
        # add axis labels
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    elif dim == 3:  # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot the recipies for each group, highest frequency first
    for i, recipe in enumerate(recipes_by_region):
        ax.plot(*zip(*recipe), marker='.', linestyle='', 
                color=colors[regions_list[i]], ms=4, label=regions_list[i])
    ax.legend(loc=4, markerscale=3, prop={'size':6})

    # Save image
    plt.savefig('figures/' + model_name + '_biplot.png')

def roc_plot(y_test, y_pred, model_name):
    """ Calculate the ROC-curve for the ingredient reconstruction and plot"""

    # Get data in right format
    test_values = np.array([item for sublist in y_test for item in sublist])
    test_scores = np.array([item for sublist in y_pred for item in sublist])

    # plot
    fpr, tpr, thresholds = roc_curve(test_values, test_scores)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.savefig('figures/' + model_name + '_roc.png')

def knn_par_plot(neighbors_list, accuracy_list, model_name):
    plt.figure()
    plt.plot(neighbors_list, accuracy_list)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of neighbors')
    plt.savefig('figures/' + model_name + '_par_knn.png')

def rankplot(ranks, add_remove='add'):
    assert add_remove in ('add', 'remove')
    rank_range = 10 if add_remove == 'add' else 15
    freq_dict = dict()
    for rank in ranks:
        if rank <= rank_range: 
            freq_dict[rank] = freq_dict.get(rank, 0) + 1
    ranks_pct = np.array(list(freq_dict.values()))/len(ranks)*100
    x = np.array(list(range(rank_range))) + 0.5
    plt.figure()
    if add_remove == 'remove':
        plt.axis([0.5, 15.5, 0, 22])
    else:
        plt.axis([0.5, 10.5, 0, 85])
    plt.bar(x, ranks_pct, 1, color="green")
    plt.xlabel('Rank')
    plt.ylabel('Test recipes (%)')
    plt.xticks(range(1, rank_range + 1))
