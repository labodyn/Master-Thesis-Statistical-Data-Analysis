#############################################################
#Project: Thesis for master in the statistical data analysis#
#Author: Lander Bodyn                                       #
#Date: September 2016                                       #
#Email: bodyn.lander@gmail.com                              #
#############################################################

#### MODULE: MAIN.PY ####

#Import train_network code
from traincode import train_network

#List to keep track of convergence times of the gradient descent under different parameters
times = []

#Execute the program several times, looping over different parameters
batchsizes = [8]
deltas = [0.25]
alphalist = [0.9]
for bs in batchsizes:
	for dt in deltas:
		for alph in alphalist:
			times.append(train_network(batch_size_ = bs,delta_ = (dt*bs), alpha_ = alph))

#Print out the convergence times. This can give a good idea for the optimal parameters.
print('Convergence times for the different parameter settings:')
print(times)



#-------------------------------END OF CODE-------------------------------#

#-------------------------------TO DO CODE-------------------------------#

#5 bottleneck dimensies, 
#LDA toepassen, prestatie curve vs dimensies
#nearest neighbour

#wat kan ik er uit leren? -> eens pracktisch gebruiken, steek ingredienten erin die logisch zijn, kijk naar output.

#pretraining: network opslaan, initialiseren met netwerk.


#pair plot all neurons. 


#zet cut off op output. (endlayer) -> maak binair, geef andere output voor hoe goed network werkt. ROC curve, false positives, false negatives, ... 

#Logische recepten output? Andere labels dan regions. bijv chocolade, kruidig, ... 


#------------------------------- GAINED KNOWLEDGE AND INTUITION -----------------------

#### Costfunction intuition

#waarom geen linear? om niet lineariteit te hebben. Lineare layers collapsen tot zero hidden layers
#CE: bestraft heel hard (logarithmisch) fouten
#linrec: handig want je kunt altijd uit leren. Sigmoid klein of groot: grad =~0, zinloos om te leren. 

#### Convergence intuition

#algorithm for fast convergence with parameter
#first: biases, ingredient frequency. second, correlations
#converged: no correlations of neurons
#first first neurons learning, than second, ... 


#### Convergence parameter intuition

#Stepsize "delta": 
#too large: initial fast convergence, but a lot of fluctuation and bad convergence when algorithm goes on.
#too small: precise but too slow convergence
#if batch size decreases, should decrease similar ratio (or bit less)

#Batch size: 
#too large batch size: very slow update, but very precise
#too small batch size: too little information per update, takes longer to converge. In combination with a too big step size, this will lead to overshooting the costfunction minima, thus worse convergence when close to the target.

#### Optimal convergence parameters 

#2,2.8,0.9 : 0.0555, 0.0551
#4,5.6,0.9 : 

#5 layer: 2 2.8
#3 layer: 6 9


