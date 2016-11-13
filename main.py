#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import time
import pandas as pd

from functions import *
from train import predict_zero, perform_svd, train_autoencoder
from plot import cost_function_plot, biplot

class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self

def make_networkstring(network):
    string = 'b' if network.has_bias else 'n'
    cfn = network.costfn.__name__.split('_')
    string += cfn[0][0] + cfn[1][0]
    string += network.actfn[0].__name__[0]
    for i, neuron in enumerate(network.n_hidden_neurons):
        string += str(neuron)
        string += network.actfn[i+1].__name__[0]
    network.string = string

def get_parameters():

    #Network settings
    network = Bunch()
    network.has_bias = True #Add bias to the layers or not
    network.costfn = cross_entropy #Costfunction
    network.actfn = [rectifier, rectifier, linear, rectifier, rectifier, sigmoid] #Activation functions between the layers
    #Number of neurons in the hidden layers
    network.n_hidden_neurons = [100, 100, 2, 100, 100] 
    make_networkstring(network)

    #Gradient descent settings
    graddesc = Bunch()
    graddesc.test_size = 0.3
    graddesc.eps_init = 0.1 #Half of the init range of the network param
    graddesc.batch_size = 8 #Number of observations to use for gradient descent
    graddesc.alpha = 0.9 #Inertie coefficient. alpha = 1 means no momentum
    graddesc.delta = 2 #Learning rate
    graddesc.max_loops = 100 #Maximum number of loops over all the data
    graddesc.max_time = 20000 #Maximum time
    graddesc.low_cost = 0.055 #Lowest cost
    graddesc.cost_update_size = 5000 #Number of observations to use before updating cost
    graddesc.plot_svd = True #Plot the cost corresponding with an SVD with the same dimension reduction as the network on the costfunction plot

    return network, graddesc

''' def output_parameters(network.string):
    #Print out the defining variables of the network
    print('-'*40)
    print("DATA       Training:", ntrain, "Testing:", ntest)
    print("NETWORK   ",network.string)
    print("GRAD DESC  Init Range:", epsilon_init , ", Batch size:",⋅
        batch_size , ", Delta:",,udelta, ", Alpha:", alpha) '''

def search_grid():

    #Get parameters
    network, graddesc = get_parameters()

    #Import data
    df = pd.read_csv('datasets/ReceptenBinair_minimun2ingredients.csv',\
            sep=';', index_col=0)
    data = df.values
    regions = df.index
    ingredient_names = df.columns
    print('Read in datafile with {} recipies and {} ingredients.'
            .format(len(data), len(ingredient_names)))

    #Train the network
    results = train_autoencoder(data, network, graddesc)
    bottleneck_neurons, cost_train, cost_test, running_time = results

    #Try out some linear reduction methods to compare with
    cost_zero = predict_zero(data, network.costfn)
    cost_svd, data_svd_reduced = perform_svd(data, 2, network.costfn)

    #Visualize results
    cost_function_plot(cost_train, cost_test, network.costfn, 
            graddesc.cost_update_size, cost_zero=cost_zero, 
            cost_svd=cost_svd) 
    input('')

    biplot(bottleneck_neurons, regions)

    input('')

    biplot(data_svd_reduced, regions)

    input('')

    #print('Cuisine regions:')
    #for region in regions_unique:
    #    print(region, end=' ')

    #Define grid
    batchsize_grid = []

    #List to store the convergence times
    times = []


if __name__ == '__main__':
    search_grid()



'''
plt.ion()
plt.ioff()


import os
import shutil


    #Make directory to save outputs
        #directory = "figures/biplot_learning/" + network.string + "/"
            #if os.path.exists(directory):
                #    shutil.rmtree(directory)
                    #os.makedirs(directory)

        #Plot the cost function
        print("Making cost function plot")
        costplot(costlist_train,costlist_test,x_train,costfn,nlowneurons,plot_svd)
        ### plt.show()
        ### wait = input("PRESS ENTER TO CONTINUE")
        plt.savefig('figures/cost/' + network_str + '.png')
        plt.close("all")

        #Make biplot of the hiddenlayer of test data
        print("Making biplot")
        biplot(bottleneck_neuron(x_test),region_test)
        ### plt.show()
        ### wait = input("PRESS ENTER TO CONTINUE")
        plt.savefig('figures/biplot/test_'+ network_str +'.png')
        plt.close("all")

        #Make biplot of the hiddenlayer of all data
        print("Making biplot")
        biplot(bottleneck_neuron(df.values),df.index)
        ### plt.show()
        ### wait = input("PRESS ENTER TO CONTINUE")
        plt.savefig('figures/biplot/all_'+ network_str +'.png')
        plt.close("all")

        #Print out neurons of the bottleneck hidden layer + region, of all
        #observations.
        print("Writing out hidden neuron values")
        hidlayoutput = bottleneck_neuron(df.values)
        groupoutput = df.index
        outputfile = open('output/neuronvalues.txt', 'w')
        outputfile.write('neuron1\tneuron2\tgroup\n')
        for i in range(len(groupoutput)):
                outputfile.write(str(hidlayoutput[i][0])+'\t'+str(hidlayoutput[i][1])+'\t')
                outputfile.write(groupoutput[i]+'\n')

        outputfile.close()
'''
