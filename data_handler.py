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

def get_complex_data(n, ssl=False):
    n_batches=2
    n_pts_per_cluster=100
    m = (n_pts_per_cluster*n)
    l = int(m*0.2)

    make = lambda x,y,s: np.concatenate([np.random.normal(x,s, (n_pts_per_cluster, 1)), np.random.normal(y,s, (n_pts_per_cluster, 1))], axis=1)
    # batch 1
    xb1_list = [make(-1.3, 2.2, .1), make(.1, 1.8, .1), make(.8, 2, .1),make(-3.2, -1, .1), make(.5, 2.3, .1), make(.1, -0.9, .1)]
    xb2_list = [make(-.9, -2, .1), make(0, -2.3, .1), make(1.5, -1.5, .1),make(4.1, 3.2, .1), make(0.1, -3.1, .1), make(-1.1, -0.8, .1)]
    
    xb1 = np.concatenate(xb1_list[:n], axis=0)
    labels1 = np.concatenate([x* np.ones(n_pts_per_cluster) for x in range(n)], axis=0)
    xb1_full = np.hstack((xb1, labels1.reshape(m,1)))

    # batch 2
    xb2 = np.concatenate(xb2_list[:n], axis=0)
    labels2 = labels1
    xb2_full = np.hstack((xb2, labels2.reshape(m,1)))

    xb1_full = np.take(xb1_full,np.random.permutation(xb1_full.shape[0]),axis=0,out=xb1_full)
    xb2_full = np.take(xb2_full,np.random.permutation(xb2_full.shape[0]),axis=0,out=xb2_full)

    if ssl:
        return xb1_full[:l,:-1], xb2_full[:l,:-1], xb1_full[:l,-1], xb2_full[:l,-1], xb1_full[l:,:-1], xb2_full[l:,:-1], xb1_full[l:,-1], xb2_full[l:,-1]
    else:
        return xb1_full[:,:-1], xb2_full[:,:-1], xb1_full[:,-1], xb2_full[:,-1]
    


def get_data(dataset, dim=5, mnili=[3 ,7], ssl=False, noise=True):
    if dataset == "glove":
        fp_intersect_word_image = "assets/intersect_glove.840b-openimage.box.p"

        intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
        z_word = intersect_data['z_0']  # The word embedding.
        z_image = intersect_data['z_1']  # The image embedding.
        vocab_intersect = intersect_data['vocab_intersect']  # The concept labels. 
        n_item = len(vocab_intersect)
        l = int(n_item*0.2)

        embedding = SpectralEmbedding(n_components=dim)
        z_word_tr = embedding.fit_transform(z_word)
        z_image_tr = embedding.fit_transform(z_image)

        lword = np.arange(len(z_word_tr)).reshape((len(z_word_tr),1))
        limage = np.arange(len(z_image)).reshape((len(z_image),1))

        zword = np.hstack((z_word_tr, lword))
        zimage = np.hstack((z_image_tr, lword))
        zword = np.take(zword,np.random.permutation(zword.shape[0]),axis=0,out=zword)
        zimage = np.take(zimage,np.random.permutation(zimage.shape[0]),axis=0,out=zimage)

        if ssl:
            return zword[:l,:-1], zimage[:l,:-1], zword[:l,-1], zimage[:l,-1], zword[l:,:-1], zimage[l:,:-1], zword[l:,-1], zimage[l:,-1]
        else:
            return zword[:,:-1], zimage[:,:-1], zword[:,-1], zimage[:,-1]
    
    
    elif dataset == "gauss":   
        n_batches=2
        n_pts_per_cluster=100
        l = int((n_pts_per_cluster*3)*0.2)
        
        make = lambda x,y,s: np.concatenate([np.random.normal(x,s, (n_pts_per_cluster, 1)), np.random.normal(y,s, (n_pts_per_cluster, 1))], axis=1)
        # batch 1
        xb1 = np.concatenate([make(-1.3, 2.2, .1), make(.1, 1.8, .1), make(.8, 2, .1)], axis=0)
        labels1 = np.concatenate([0 * np.ones(n_pts_per_cluster), 1 * np.ones(n_pts_per_cluster), 2 * np.ones(n_pts_per_cluster)], axis=0)
        xb1_full = np.hstack((xb1, labels1.reshape(300,1)))
        
        # batch 2
        xb2 = np.concatenate([make(-.9, -2, .1), make(0, -2.3, .1), make(1.5, -1.5, .1)], axis=0)
        labels2 = np.concatenate([0 * np.ones(n_pts_per_cluster), 1 * np.ones(n_pts_per_cluster), 2 * np.ones(n_pts_per_cluster)], axis=0)
        xb2_full = np.hstack((xb2, labels2.reshape(300,1)))
        
        xb1_full = np.take(xb1_full,np.random.permutation(xb1_full.shape[0]),axis=0,out=xb1_full)
        xb2_full = np.take(xb2_full,np.random.permutation(xb2_full.shape[0]),axis=0,out=xb2_full)
        
        if ssl:
            return xb1_full[:l,:-1], xb2_full[:l,:-1], xb1_full[:l,-1], xb2_full[:l,-1], xb1_full[l:,:-1], xb2_full[l:,:-1], xb1_full[l:,-1], xb2_full[l:,-1]
        else:
            return xb1_full[:,:-1], xb2_full[:,:-1], xb1_full[:,-1], xb2_full[:,-1]
        
    elif dataset == "mnist":   
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        images,labels = mnist.train.images, mnist.train.labels
        labels = np.asarray([[labels[i],i] for i in range(len(mnili)*200) if (labels[i] in mnili)])
        images = images[labels[:,1]]
        images = images.reshape((-1, 28, 28))

        xb1, labels1 = images, labels
        labels2 = labels1
        
        xb2 = np.zeros(xb1.shape)
        for x in range(len(xb1)):
            xb2[x] = np.rot90(xb1[x])
            
        if noise:
            xb1 = xb1+np.random.normal(0,0.1,xb1.shape)
        
        l = len(labels)//2

        if ssl:
            return xb1[:l], xb2[:l], labels1[:l], labels2[:l], xb1[l:], xb2[l:], labels1[l:], labels2[l:]
        else:
            return xb1, xb2, labels1, labels2
        

        
    else:
        raise ValueError("Please enter {gauss, glove or mnist}")