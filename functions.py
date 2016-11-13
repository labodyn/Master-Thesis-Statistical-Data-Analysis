#####################################################
#Project: Master Thesis in Computational Statistics #
#Author: Lander Bodyn                               #
#Date: January 2017                                 #
#Email: bodyn.lander@gmail.com                      #
#####################################################

import theano as th
import theano.tensor as T
from numpy import random as r

#Define the types of activation functions
def linear(x, w, b):
    return T.dot(x, w) + b

def rectifier(x, w, b):
    return T.maximum(0, T.dot(x, w) + b)

def sigmoid(x, w, b):
    return T.nnet.sigmoid(T.dot(x, w) + b)

#Define the types of cost functions
def least_squares(output, target):
    return T.mean((output - target)**2)

def cross_entropy(output, target):
    return T.nnet.binary_crossentropy(output, target).mean()

#Define the initialisation of the network parameters
def init_weight(dim1, dim2, eps_init):
    return th.shared((r.uniform(size=(dim1, dim2))-0.5)*eps_init)

def init_bias(dim, eps_init):
    return th.shared((r.uniform(size=(dim))-0.5)*eps_init)
