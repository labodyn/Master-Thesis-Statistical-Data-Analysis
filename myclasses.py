#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

from pprint import pprint

class NNParameters():
    '''Class to keep track of the parameters in an orderly manner.'''

    def __init__(self, par_dict):

        # Set values in par_dict as class attributes
        for key, value in par_dict.items():
            setattr(self, key, value)

        assert len(self.n_hidden_neurons) + 1 == len(self.activation_fn)
        self.make_other_parameters()

    def __str__(self):
        '''Define how to represent the network parameters as string'''

        string = 'b' if self.has_bias else 'n'
        cfn = self.loss_fn.__name__.split('_')
        string += cfn[0][0] + cfn[1][0]
        string += self.activation_fn[0].__name__[0]
        for i, neuron in enumerate(self.n_hidden_neurons):
            string += str(neuron)
            string += self.activation_fn[i+1].__name__[0]
        return string

    def make_other_parameters(self):
        '''Make some other relevant parameters'''

        #Determine the index of the smallest layer in the full network
        neuron_min = min(self.n_hidden_neurons)
        self.bottleneck_index = self.n_hidden_neurons.index(neuron_min) + 1

    def print_out(self):
        print('-'*79)
        pprint(vars(self))
        print('-'*79)
