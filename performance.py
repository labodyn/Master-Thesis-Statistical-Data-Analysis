#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import numpy as np
from statistics import mean, median

# Import from own modules
from train_other import train_knn, train_qda
from plot import biplot, rankplot
from get_data import get_data

def predict_regions(give_neurons, p, model_name):
    """ Predict the regions of the recipes using the neurons of the bottleneck
    layer in the autoencoder. Filter out NorthAmerican for region prediction: 
    NorthAmerican contains recipes of all regions and dominates in numbers, 
    which makes the prediction trivial. Since the training of the autoencoder
    was unsupervised, we can use all data for this part"""

    # Get the data (without North American recipes)
    data, regions, ingredients = get_data(remove_northamerican=True)

    # Get the reduced features
    all_neurons = give_neurons(data.tolist())
    bottleneck_neurons = all_neurons[p.bottleneck_index]
    #bottleneck_neurons = np.dot(data, give_neurons)

    # Split in 80% train and 20% test data
    n_train = int(len(regions) * 0.8)
    data_train = bottleneck_neurons[:n_train]
    data_test = bottleneck_neurons[n_train:]
    regions_train = regions[:n_train]
    regions_test = regions[n_train:]

    # KNN
    regions_predicted_knn = train_knn(data_train, data_test, regions_train, 
            regions_test, model_name)

    # QDA
    regions_predicted_qda = train_qda(data_train, data_test, regions_train, 
            regions_test)

    # Make a biplot of the train, test and predictions
    biplot(data_train, regions_train, model_name + '_train')
    biplot(data_test, regions_test, model_name + '_test')
    biplot(data_test, regions_predicted_knn, model_name + '_pred_knn')
    biplot(data_test, regions_predicted_qda, model_name + '_pred_qda')

def make_recipes(data, give_neurons, add_remove, ingredients=None, show=0):
    """ Testing the autoencoder network as recommender system. Using a dataset
    that has not been used to train the autoencoder, either add or remove an
    ingredient. Measure the reconstruction rank of the ingredient in the list 
    of ingredients (add) or the list of unused ingredients (remove). """

    # This will increase the accuracy of the performance measures by repeating
    # the procedure 10x for the whole dataset.
    loops = 20

    assert add_remove in ('add', 'remove')
    par = 0 if add_remove == 'remove' else 1

    rank_list = []
    show_counter = 1
    for _ in range(loops):
        for recipe_orig in data:

            # Don't chance original recipes
            recipe = recipe_orig.copy()

            # Select ingredients and remove/add one randomly
            ingr_idxs = np.argwhere(recipe == 1).flatten()
            idxs = np.argwhere(recipe != par).flatten()
            idx = np.random.choice(idxs)
            recipe[idx] = par

            # Get output layer of the autoencoder (needs 2D array as input)
            reconstruction = give_neurons(recipe.reshape(1,-1))[-1][0]

            # Set the ingredients we don't want in the ranking to a value that 
            # will come last.
            reconstruction[recipe != par] = -1 + 3*par

            # Get the ingredient indexes sorted on reconstruction
            reconstruction_idxs = zip(reconstruction, range(len(recipe)))
            sorted_reconstr_idxs = sorted(reconstruction_idxs, reverse=not par)
            _, sorted_idxs = zip(*sorted_reconstr_idxs)

            # Get index of removed ingredient, add 1 for the rank.
            rank = sorted_idxs.index(idx) + 1
            rank_list.append(rank)

            # If show, print out show recipes with reconstruction in top 6 and
            # maximum 6 ingredients
            if show_counter <= show and rank <= 6 and len(ingr_idxs) <= 6:
                assert ingredients is not None, 'Ingredients must be given to show'
                show_counter += 1
                print('Original: ' + ' | '.join(ingredients[i] for i in ingr_idxs))
                print('{} ingredient: {}'.format(add_remove, ingredients[idx]))
                print('Top 10 reconstruction: ')
                ingredient_counter = 0
                for reco, sorted_idx in sorted_reconstr_idxs:
                    ingredient_counter += 1
                    print('{: <25}: {:1.4f}'.format(ingredients[sorted_idx], reco))
                    if ingredient_counter > 6:
                        print('-'*79)
                        break

    # Calculate measures of performance
    mean_rank = mean(rank_list) 
    median_rank = median(rank_list)
    top1 = mean(int(rank == 1) for rank in rank_list) * 100
    top10 = mean(int(rank <= 10) for rank in rank_list) * 100
    out_str = ('{:6} | mean rank: {:6.2f}, median rank: {:5.1f}, '
            'top1: {:6.2f}%, top10: {:6.2f}%' + '\n'*int(bool(show)))
    print(out_str.format(add_remove, mean_rank, median_rank, top1, top10))

    # Make plot of ranks
    plot = True
    if plot:
        rankplot(rank_list)

    return mean_rank
