#############################################################
#Project: Thesis for master in the statistical data analysis#
#Author: Lander Bodyn                                       #
#Date: September 2016                                       #
#Email: bodyn.lander@gmail.com                              #
#############################################################

#### MODULE: PLOTCODE.PY ####

import numpy as np
from numpy import random as r
import matplotlib.pyplot as plt
import math
plt.ion()
from numpy import genfromtxt
import pandas
import matplotlib.patches as mpatches
import statistics as st
from collections import Counter
from itertools import groupby
from cycler import cycler

#Only show plot when asked
plt.ioff()

def costplot(costlist_train,costlist_test,x_train,costfn,nlowneurons,plot_SVD):

	#Plot cost function
	plot1, =plt.plot(costlist_train,'ro')
	plot2, =plt.plot(costlist_test,'go')
	plt.ylabel('Cost')
	plt.xlabel('Number of iterations')
	plt.title('Costfunction with ' + str( nlowneurons) + ' hidden neurons')
	plt.legend([plot1,plot2], ['Training batch average', 'Testing'])

	#Plot SVD cost and predict-zero cost on costfunction
	if(plot_SVD):
		#Calculating SVD cost with equivalent dimension reduction
		U, D, VT = np.linalg.svd(x_train, full_matrices=False)
		V_k = VT.T[:,0:nlowneurons]
		x_k = np.dot(np.dot(x_train,V_k),V_k.T)
		if costfn == 'ls':
			cost2 = np.mean((x_k-x_train)**2)
		if costfn == 'ce':
			costmatrix2 = -(x_train*np.log(x_k)+(1-x_train)*np.log(1-x_k))
			costmatrix2[np.isnan(costmatrix2)] = 0
			cost2 = np.mean(costmatrix2)
		print("Cost SVD:", cost2)
		#Plot svd line
		plt.axhline(y=cost2, xmin=0, xmax=1, hold=None)
		plt.annotate('Cost SVD', (5,cost2))
		#Calculating predict-zero cost
		if costfn == 'ls':
			cost3 = np.mean((x_train)**2)
		if costfn == 'ce':
			costmatrix3 = -x_train*np.log(0)
			costmatrix3[np.isnan(costmatrix3)] = 0
			cost3 = np.mean(costmatrix3)
		print("Cost predict-zero:", cost3)
		#plot predict-zero line
		plt.axhline(y=cost3, xmin=0, xmax=1, hold=None)
		plt.annotate('Cost predict-zero', (5,cost3))


def biplot(hiddenlayer,regions):

	#Make dataframe of the hidden layer
	xs = [x[0] for x in hiddenlayer]
	ys = [x[1] for x in hiddenlayer]
	dfbiplot = pandas.DataFrame(dict(x=xs, y=ys, region=regions))

	#Make region a categorical variable sorted descending on region frequency
	regionfreq_test = [[len(list(g)),k] for k, g in groupby(sorted(regions))]
	regionfreq_test.sort(reverse=True)
	sortedregions_test = [x[1] for x in regionfreq_test]
	dfbiplot['region'] = pandas.Categorical(dfbiplot['region'], sortedregions_test)
	dfbiplot = dfbiplot.sort_values(['region'])

	# Plot recipes on the first two hidden neurons, colored by region
	fig, ax = plt.subplots()
	ax.set_prop_cycle(cycler('color', ['lightgray', 'purple', 'red', 'violet','orange','green', 'greenyellow','yellow', 'blue','black', 'turquoise']))
	###ax.set_color_cycle(['lightgray', 'purple', 'red','yellow', 'blue','green', 'yellow','yellow', 'blue','black', 'blue'])
	###sortedregions_test = ['NorthAmerican', 'SouthernEuropean', 'LatinAmerican', 'EastAsian', 'WesternEuropean', 'MiddleEastern', 'SouthAsian', 'SoutheastAsian', 'NorthernEuropean', 'African', 'EasternEuropean']
	groups = dfbiplot.groupby('region')
	for name, group in groups:
		ax.plot(group.x, group.y, marker='.', linestyle='', ms=3, label=name)

	ax.legend(loc=4,markerscale = 3,prop={'size':6})



