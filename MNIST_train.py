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
from data_handler import *
import joblib
# change model here
from python_models.procrustes import *

"""
This is a pilot test to run the MNIST experiment. Import the model and change accordingly.
"""

def mnist_simple(plot=True):
	results = []
	ali = []
	for i in range(100):
	    xb1_train, xb2_train, labels1_train, labels2_train, xb1_test, xb2_test, labels1_test, labels2_test = get_data("mnist", ssl=True, noise=False)
	    xb1_train, xb2_train = xb1_train.reshape(-1, 28* 28), xb2_train.reshape(-1, 28* 28)
	    Q, k, X_mean, Y_mean = Procrustes(xb1_train.T, xb2_train.T)

	    Y_u_2 = k*xb2_test.reshape(-1, 28* 28)@Q
	    Y_u_1 = k*xb1_test.reshape(-1, 28* 28)@Q
	    acc = 0
	    for i in range(len(Y_u_1)):
	        dists = []
	        for j in range(len(Y_u_2)):
	            dist = np.linalg.norm(Y_u_2[j]-Y_u_1[i])
	            dists.append(dist)
	        if labels1_test[i,0] == labels2_test[np.argmin(dists),0]:
	            acc +=1

	    results.append(acc/len(Y_u_1)) 
	      
	print("mapping accuracy simple procrustes: %f%%"%(np.mean(results)*100))

	if plot:
		import matplotlib.pyplot as plt
		fig, axes = plt.subplots(2, 2, sharey=True)
		axes[0, 0].set_title('Original')
		axes[0, 1].set_title('Generated')

		axes[0,0].imshow(xb1_test[10].reshape(28,28))
		axes[1,0].imshow(xb2_test[10].reshape(28,28))

		axes[0,1].imshow(Y_u_1[10].reshape(28,28))
		axes[1,1].imshow(Y_u_2[10].reshape(28,28))

		plt.show()

		

def mnist_complex(save=True):
	procrustes_metrics = []

	for mnili in [np.arange(10)[:i] for i in range(2,11)]:
	    print("***dataset***", mnili)
	    results = []
	    ali = []
	    for i in range(10):
	        xb1_train, xb2_train, labels1_train, labels2_train, xb1_test, xb2_test, labels1_test, labels2_test = get_data("mnist", mnili=mnili, ssl=True, noise=False)
	        xb1_train, xb2_train = xb1_train.reshape(-1, 28* 28), xb2_train.reshape(-1, 28* 28)
	        Q, k, X_mean, Y_mean = Procrustes(xb1_train.T, xb2_train.T)

	        Y_u_2 = k*xb2_test.reshape(-1, 28* 28)@Q
	        Y_u_1 = k*xb1_test.reshape(-1, 28* 28)@Q
	        acc = 0
	        for i in range(len(Y_u_1)):
	            dists = []
	            for j in range(len(Y_u_2)):
	                dist = np.linalg.norm(Y_u_2[j]-Y_u_1[i])
	                dists.append(dist)
	            if labels1_test[i,0] == labels2_test[np.argmin(dists),0]:
	                acc +=1

	        results.append(acc/len(Y_u_1)) 

	    print("mapping accuracy complex procrustes: %f%%"%(np.mean(results)*100))
	    procrustes_metrics.append([np.mean(results)])

if save:
    	filename = "results/mnist/mnist_complex_procrustes.p"
		joblib.dump(procrustes_metrics, filename, compress=9)



if __name__ == "__main__":
	mnist_simple(False)
	mnist_complex(False)



