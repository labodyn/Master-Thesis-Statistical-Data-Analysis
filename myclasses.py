#Class to keep track of the parameters in an orderly manner.
from collections import OrderedDict

class NNParameters(OrderedDict):

    def __init__(self, *args, **kwds):
        super(NNParameters, self).__init__(*args, **kwds)
        for key, value in self.items():
            setattr(self, key, value)
        #self.__dict__ = self

        try:
            self.make_network_string()
        except:
            print('Parameters missing in the construction of NNParameters!!')


    def make_network_string(self):
        '''Make a string of the network parameters'''

        self.string = 'b' if self.has_bias else 'n'
        cfn = self.cost_fn.__name__.split('_')
        self.string += cfn[0][0] + cfn[1][0]
        self.string += self.activation_fn[0].__name__[0]
        for i, neuron in enumerate(self.n_hidden_neurons):
            self.string += str(neuron)
            self.string += self.activation_fn[i+1].__name__[0]

    def print_out(self):

        def out(element):
            return element.__name__ if callable(element) else str(element)

        print('-'*34 + 'PARAMETERS' +'-'*34)
        for key, value in self.items():
            if type(value) == list:
                print('{:<17}: '.format(key) + ', '.join(out(el) 
                    for el in value))
            else:
                print('{:<17}:'.format(key), out(value))
        print('-'*78)
