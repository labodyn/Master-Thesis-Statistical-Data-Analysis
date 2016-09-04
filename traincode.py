#############################################################
#Project: Thesis for master in the statistical data analysis#
#Author: Lander Bodyn                                       #
#Date: September 2016                                       #
#Email: bodyn.lander@gmail.com                              #
#############################################################

#### MODULE: TRAINCODE.PY ####

#Import packages
import theano as th
import theano.tensor as T
import numpy as np
from numpy import random as r
import matplotlib.pyplot as plt
import math
plt.ion()
from numpy import genfromtxt
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.patches as mpatches
import statistics as st
from collections import Counter
from itertools import groupby
import time
import os
import shutil

#Import own functions
from plotcode import costplot
from plotcode import biplot

#Block pop up images
plt.ioff()

#Define the types of activation functions
def ln(x,w,b):
	return T.dot(x,w) + b
def lr(x,w,b):
	return T.maximum(0,T.dot(x,w) + b)
def sm(x,w,b):
	return T.nnet.sigmoid(T.dot(x,w) + b)

def train_network(batch_size_ = 10, delta_ = 1.25, alpha_ = 0.9):

	#### SETTINGS ####

	#Network options
	costfn = 'ce' #least squares 'ls', crossentropy 'ce'
	actfn = [lr,lr,ln,lr,lr,sm] #linear (ln) , linear rectifier (lr), sigmoid (sm)
	nhidlayneur = [100,100,2,100,100] #The number of neurons in the hidden layers. 5 a 10 for middle layer
	add_bias = True #Add biases or not. Biases to learn basic frequencies.

	#Plot parameters
	plot_SVD = True #Plot the cost corresponding with an SVD with the same dimension reduction as the network on the costfunction plot

	#Gradient descent parameters
	epsilon_init = 0.1 #initalisation range of network parameters

	#Regularisatieparameters

	#Minibatch variables
	batch_size = batch_size_ #Size of minibatch
	batch_plotav = 10000 #Number of observations to average over for costfunction plot.

	#Momentum parameters
	alpha = alpha_ #Inertie coefficient. alpha = 1 means no momentum (but still 1 delay for update)
	delta = delta_ #0.25 #Learning rate

	#Stopping criteria (experimental).
	#Converged when all these true
	ndata_min = 3 #minimum output points for cost function plot
	#And one of these true
	nloops_max = 100 #Loops over all the data
	cost_conv = 0.065 #Test cost
	time_max_sec = 20000 #Maximum convergence time

	#### CODE ####

	#Start counting program execution time
	start = time.time()

	#Import dataset
	df = pd.read_csv('datasets/ReceptenBinair_minimun2ingredients.csv', sep=';',index_col=0)

	#Data exploration
	col_headers = list(df.columns)
	row_headers = list(df.index)
	nvar = len(col_headers)#381 ingredients
	nobs = len(row_headers)#55001 recipes
	counts = Counter(row_headers)
	region_list = set(row_headers)
	nregion = len(region_list)#11 regions

	#Split data + randomize
	data_train, data_test = train_test_split(df, test_size=0.3, random_state=42) 
	x_train = data_train.values
	x_test = data_test.values
	region_train = list(data_train.index)
	region_test = list(data_test.index)
	ntrain = len(x_train) #44000 training data
	ntest = len(x_test) #11001 test data

	#Create the mini-batches (works, but ugly code)
	numberofbatches = int(len(x_train)/batch_size) #put the rest of the quotient together with the last batch so we have no small batches
	train_batches = []
	for i in range(numberofbatches):
		x_batch = x_train[(i*batch_size):((i+1)*batch_size),:]
		if (i==numberofbatches-1):
			x_batch = x_train[((numberofbatches-1)*batch_size):len(x_train),:]
		train_batches.append(x_batch)
	
	#Define some variables related to the number of neurons in each layer
	nlayneur = [nvar] + nhidlayneur + [nvar]
	nlay = len(nlayneur)
	nlowneurons = min(nhidlayneur)
	laylowneuron = nlayneur.index(nlowneurons)
	
	#Create string defining the network. Usefull for plot names
	networkstring = str(actfn[0]).split()[1]
	for i in range(len(nhidlayneur)):
		networkstring+= '_' + str(nhidlayneur[i])
		networkstring+= '_' + str(actfn[i+1]).split()[1]
	networkstring+= ':' + costfn
	if add_bias:
		networkstring+= ':b'
	
	#Print out the defining variables of the network
	print("------------------------------------------------------------------")
	print("DATA       Training:", ntrain, "Testing:", ntest)
	print("NETWORK   ",networkstring)
	print("GRAD DESC  Init Range:", epsilon_init , ", Batch size:", batch_size , ", Delta:", delta, ", Alpha:", alpha)
	
	#Define arrays over the layers of the theano variables.
	t = T.dmatrix()
	x = [T.dmatrix()]
	w = []
	b = []
	grad_w = []
	grad_b = []
	dg_w = []
	dg_b = []
	
	#Loop over other layers
	for i in range(nlay-1):
		w.append(th.shared((r.uniform(size=(nlayneur[i],nlayneur[i+1]))-0.5)*epsilon_init))
		b.append(th.shared((r.uniform(size=(nlayneur[i+1]))-0.5)*epsilon_init))
		grad_w.append(th.shared((r.uniform(size=(nlayneur[i],nlayneur[i+1]))-0.5)*epsilon_init))
		grad_b.append(th.shared((r.uniform(size=(nlayneur[i+1]))-0.5)*epsilon_init))
		x.append(actfn[i](x[i],w[i],b[i]))
	
	#Costfunction compairing outcome with last layer. For binairy data, crossentropy is a more logical costfunction
	if costfn == 'ls':
		cost = T.mean((x[len(x)-1]-t)**2)
	if costfn == 'ce':
		cost = T.nnet.binary_crossentropy(x[len(x)-1], t).mean()
	
	#Calculate the gradient
	for i in range(nlay-1):
		dg_w.append(T.grad(cost, w[i]))
		dg_b.append(T.grad(cost, b[i]))
	
	#Define how to update the parameters
	updates_w = [(w[i], w[i]-delta*grad_w[i]) for i in range(nlay-1)]
	updates_b = [(b[i], b[i]-delta*grad_b[i]) for i in range(nlay-1)]
	updates_w_momentum = [(grad_w[i], grad_w[i]-alpha*(grad_w[i]-dg_w[i])) for i in range(nlay-1)]
	updates_b_momentum = [(grad_b[i], grad_b[i]-alpha*(grad_b[i]-dg_b[i])) for i in range(nlay-1)]
	updates = updates_w  + updates_w_momentum
	if add_bias: 
		updates += updates_b + updates_b_momentum
	
	
	# param -> parameter + rate*dgrad
	# dgrad -> grad*(1-alpha) + alpha*dgrad
	
	
	
	#Compile Theano functions
	train = th.function([x[0], t], cost, updates=updates)
	returncost = th.function([x[0], t], cost)
	returngrad = th.function([x[0], t], dg_w[0])
	returnhidlayer = th.function([x[0]], x[laylowneuron])


	#Define parameters for the gradient descent
	costlist_train_all = []
	costlist_train = [] #List with the costs of the training set
	costlist_test = [] #List with the costs of the test set
	plot_every = int(batch_plotav/batch_size)
	updatenumber = 1 
	
	#Loop over all data
	converged = False
	nloops = 0
	
	#Make directory to save outputs
	directory = "figures/biplot_learning/"+networkstring + "/"
	if os.path.exists(directory):
		shutil.rmtree(directory)

	os.makedirs(directory)

	#plot number
	plotnumber = 1

	#Perform gradient descent over all data untill converged
	while(not(converged)):
		nloops+=1	
		print("--------- data loop number", nloops,", update number",updatenumber," ---------")
	
		#biplot(returnhidlayer(x_test),region_test)
		#plt.savefig('biplotloop/'+ networkstring + str(updatenumber) +'.png')
		#plt.close("all")
	
		#Loop over minibatches
		for x_batch in train_batches:

			#cost train minibatch
			c = train(x_batch,x_batch)
			costlist_train_all.append(c)
	
			#combine number of sequential minibatches for visualisation
			if(updatenumber%plot_every==0): 
				costmean = costlist_train_all
				costmean = np.mean(costlist_train_all[updatenumber-plot_every:updatenumber])
				costlist_train.append(float(costmean))
	
				#cost test
				c2 = returncost(x_test,x_test)
				costlist_test.append(float(c2))
	
				print("Cost test:",np.round(c2, 8) ,"\tCost batch_av:", np.round(costmean, 8))
	
				#if(updatenumber>1 and costlist_test[len(costlist_test)-2]<costlist_test[len(costlist_test)-1]):
					#print("Increasing cost function. Converged?")
				
				#Make biplot for every cost update
				biplot(returnhidlayer(x_test),region_test)
				plt.savefig(directory + str(updatenumber) + "_" +str(c2) +'.png')
				#plt.savefig(directory + 'img{0:03d}'.format(plotnumber) +'.png')
				plt.close("all")
				#plotnumber += 1
	
				#Stopping criteria (experimental)
				if(len(costlist_test)>ndata_min):#at least ndata_min cost function points
				
					#stop when cost_conv is reached by test cost
					if(costlist_test[len(costlist_test)-1]<cost_conv):
						converged = True
						break
	
					#stop when number of maximum loops is reached
					if(nloops>=nloops_max):
						converged = True

					if(time.time()-start>time_max_sec):
						converged = True
						break
			
			#Update number of minibatch steps done.	
			updatenumber+=1	

	#Calculate and print out convergence time
	convergence_time = np.round((time.time()-start),1)
	print("--------- Gradient descent has converged in "+ str(convergence_time) +  " seconds ---------")


	#Plot the cost function
	print("Making cost function plot")
	costplot(costlist_train,costlist_test,x_train,costfn,nlowneurons,plot_SVD)
	### plt.show()
	### wait = input("PRESS ENTER TO CONTINUE")
	plt.savefig('figures/cost/' + networkstring + '.png')
	plt.close("all")

	#Make biplot of the hiddenlayer of test data
	print("Making biplot")
	biplot(returnhidlayer(x_test),region_test)
	### plt.show()
	### wait = input("PRESS ENTER TO CONTINUE")
	plt.savefig('figures/biplot/test_'+ networkstring +'.png')
	plt.close("all")

	#Make biplot of the hiddenlayer of all data
	print("Making biplot")
	biplot(returnhidlayer(df.values),df.index)
	### plt.show()
	### wait = input("PRESS ENTER TO CONTINUE")
	plt.savefig('figures/biplot/all_'+ networkstring +'.png')
	plt.close("all")

	#Print out neurons of the bottleneck hidden layer + region, of ALL the observations.
	print("Writing out hidden neuron values")
	hidlayoutput = returnhidlayer(df.values)
	groupoutput = df.index
	outputfile = open('output/neuronvalues.txt', 'w')
	outputfile.write('neuron1\tneuron2\tgroup\n')
	for i in range(len(groupoutput)):
		outputfile.write(str(hidlayoutput[i][0])+'\t'+str(hidlayoutput[i][1])+'\t')
		outputfile.write(groupoutput[i]+'\n')

	outputfile.close()

	#return convergence time
	return convergence_time	

