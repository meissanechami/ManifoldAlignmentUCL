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

def glove_simple():
	results = []
	ali = []
	for j in range(100):
	    xb1_train, xb2_train, labels1_train, labels2_train, xb1_test, xb2_test, labels1_test, labels2_test = get_data("glove", ssl=True, dim=5)

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
	print("mapping accuracy procrustes: %f%%"%(np.mean(results)*100))

