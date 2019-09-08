import pickle
import numpy as np
from scipy.spatial.distance import cdist,pdist, squareform
import scipy as scip
from scipy.linalg import eig,eigh
from scipy.stats import spearmanr
from math import factorial
import itertools
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf
from sklearn.manifold import Isomap, SpectralEmbedding
from numpy import genfromtxt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits import mplot3d
from python_models.procrustes import *
from data_handler import *
import joblib

"""
This is a pilot test to run the artificial experiment. Import the model and change accordingly.
"""

def artificial_simple(plot=True):
	"""
	description: simple artificial dataset experiment using procrustes
	returns: mapping accuracy and plot
	"""

	results = []
	for i in range(100):
	    xb1_train, xb2_train, labels1_train, labels2_train, xb1_test, xb2_test, labels1_test, labels2_test = get_data("gauss", ssl=True)
	    Q, k, X_mean, Y_mean = Procrustes(xb1_train.T, xb2_train.T)

	    Y_u_2 = k*xb2_test@Q
	    Y_u_1 = k*xb1_test@Q
	    acc = 0
	    for i in range(len(Y_u_1)):
	        dists = []
	        for j in range(len(Y_u_2)):
	            dist = np.linalg.norm(Y_u_2[j]-Y_u_1[i])
	            dists.append(dist)
	        if labels1_test[i] == labels2_test[np.argmin(dists)]:
	            acc +=1
	    results.append(acc/len(Y_u_1)) 
	    
	print("mapping accuracy simple procrustes: %f%%"%(np.mean(results)*100))

	if plot:
		import matplotlib.pyplot as plt
		color_map = np.asarray([["r","tomato","lightsalmon"],["navy","b","lightblue"]])
		fig, axes = plt.subplots(2, 3, sharey=True)
		axes[0, 0].set_title('Original Alignment', size=10)
		axes[0, 1].set_title('Expected Alignment', size=10)
		axes[0, 2].set_title('Generated Alignment', size=10)

		axes[0, 0].scatter(0, 0, s=45, c='r', label='Domain 1'); axes[0, 0].scatter(0,0, s=100, c='w'); #axes[0, 0].legend(handletextpad=.1, borderpad=.5, loc='lower left', bbox_to_anchor=[.02, .5]);
		axes[0, 2].scatter(0, 0, s=45, c='b', label='Domain 2'); axes[0, 2].scatter(0,0, s=100, c='w'); #axes[0, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
		axes[1, 0].scatter(0, 0, s=45, c='b', label='Domain 2'); axes[1, 0].scatter(0,0, s=100, c='w'); #axes[1, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
		axes[1, 2].scatter(0, 0, s=45, c='r', label='Domain 1'); axes[1, 2].scatter(0,0, s=100, c='w'); #axes[1, 1].legend(handletextpad=.1, borderpad=.5, loc='lower left', bbox_to_anchor=[.02, .5]);

		for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
		    axes[0, 0].scatter(xb1_test[labels1_test == lab, 0], xb1_test[labels1_test == lab, 1], s=45, alpha=.5, cmap="Reds", c=color_map[1,lab], marker=marker)
		    axes[0, 2].scatter(Y_u_1[labels1_test == lab, 0], Y_u_1[labels1_test == lab, 1], s=45, alpha=.5, cmap="Reds", c=color_map[1,lab], marker=marker)
		    axes[1, 1].scatter(xb1[labels1 == lab, 0], xb1[labels1 == lab, 1], s=45, alpha=.5, c=color_map[0,lab], marker=marker)
		    
		for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
		    axes[1, 0].scatter(xb2_test[labels2_test == lab, 0], xb2_test[labels2_test == lab, 1], s=45, alpha=.5, cmap="Reds", c=color_map[0,lab], marker=marker)
		    axes[1, 2].scatter(Y_u_2[labels2_test == lab, 0], Y_u_2[labels2_test == lab, 1], s=45, alpha=.5, cmap="Reds", c=color_map[0,lab], marker=marker)
		    axes[0, 1].scatter(xb2[labels2 == lab, 0], xb2[labels2 == lab, 1], s=45, alpha=.5,  c=color_map[1,lab], marker=marker)

		 

		axes[0, 2].legend(loc='upper left',  bbox_to_anchor=[1., .5])
		axes[1, 2].legend(loc='lower left',  bbox_to_anchor=[1., .5])


		for ax in fig.get_axes():
		    ax.label_outer() 
		    
		plt.show()


def artificial_complex(save=True):
	"""
	description: complex artificial dataset experiment using procrustes
	returns: mapping accuracy 
	"""
	procrustes_metrics = []

	for n in range(2,7):
	    print("***dataset***", n)
	    results = []
	    ali = []
	    for i in range(100):
	        xb1_train, xb2_train, labels1_train, labels2_train, xb1_test, xb2_test, labels1_test, labels2_test = get_complex_data(n,ssl=True)
	        Q, k, X_mean, Y_mean = Procrustes(xb1_train.T, xb2_train.T)

	        Y_u_2 = k*xb2_test@Q
	        Y_u_1 = k*xb1_test@Q
	        acc = 0
	        for i in range(len(Y_u_1)):
	            dists = []
	            for j in range(len(Y_u_2)):
	                dist = np.linalg.norm(Y_u_2[j]-Y_u_1[i])
	                dists.append(dist)
	            if labels1_test[i] == labels2_test[np.argmin(dists)]:
	                acc +=1
	        results.append(acc/len(Y_u_1)) 
	        

	    print("mapping accuracy complex procrustes: %f%%"%(np.mean(results)))
	    
	    procrustes_metrics.append([np.mean(results)])

	if save:
		filename = "results/artificial_complex_procrustes.p"
		joblib.dump(procrustes_metrics, filename, compress=9)

if __name__ == "__main__":
	artificial_simple(False)
	artificial_complex(False)
    





