import numpy as np
from scipy.spatial.distance import cdist,pdist, squareform
import scipy as scip
from scipy.linalg import eig,eigh
from math import factorial
import itertools
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf
from numpy import genfromtxt
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def alignment_score(a, b, method='spearman'):
    """Return the alignment score between two similarity matrices.
    
    Assumes that matrix a is the smaller matrix and crops matrix b to
    be the same shape.
    """
    n_row = a.shape[0]
    b_cropped = b[0:n_row, :]
    b_cropped = b_cropped[:, 0:n_row]
    idx_upper = np.triu_indices(n_row, 1)

    if method == 'spearman':
        # Alignment score is the Spearman correlation coefficient.
        alignment_score, _ = spearmanr(a[idx_upper], b_cropped[idx_upper])
    else:
        raise ValueError(
            "The requested method '{0}'' is not implemented.".format(method)
        )
    return alignment_score
    
def symmetric_matrix_indexing(m, perm_idx):
    """Index matrix symmetrically.
    
    Can be used to symmetrically swap both rows and columns or to
    subsample.
    """
    m_perm = copy.copy(m)
    m_perm = m_perm[perm_idx, :]
    m_perm = m_perm[:, perm_idx]
    return m_perm
    

class Greedy(object):
    """ TODO: 
        --m>1
        --number of iterations 
        --total eval = m*iter / 1*10000/100*1000
        compute new best state and build up on that
        
        next steps: check for multiple spaces procrustes: Chang Wang/Mahadevan and check references"""
    def __init__(self, n_item, n_search, s_word,rand_idx, s_image, episode, epsilon=None, paracc=False):
        """Initialise the agent."""
        self.n_search = n_search
        self.n_item = n_item
        self.s_word = s_word
        self.s_image = s_image
        self._epsilon = epsilon
        self.image_idx =rand_idx
        self._episode = episode
        self.name = 'greedy'
        self.paracc = paracc
        self.reset()

    def step(self, _step, s_word, s_image):
        action = self.getPerm(self.image_idx, _step)
        #print("current index",action)
        self._estimates[_step] = alignment_score(s_word, symmetric_matrix_indexing(s_image, action))
        #print("current _estimates",self._estimates[_step])
        best_action = np.argmax(self._estimates)  
        if self._epsilon is not None:
            if np.random.random() < self._epsilon:
                best_action = np.random.randint(self.n_item)
        #print("ali score", self._estimates[best_action])
        return best_action, self._estimates[best_action]

    def reset(self):
        self._estimates = np.zeros((self.n_search,))
        
    def getPerm(self, seq, index):
        "Returns the <index>th permutation of <seq>"
        seqc= list(seq[:])
        seqn= [seqc.pop()]
        divider= 2 # divider is meant to be len(seqn)+1, just a bit faster
        while seqc:
            index, new_index= index//divider, index%divider
            seqn.insert(new_index, seqc.pop())
            divider+= 1
        return seqn
    
    def __call__(self):
        buffer = []
        scores = []
        accuracies = []
        for e in range(self._episode):
            self.reset()
            episode_score = []
            episode_accuracy = []
            #print("episode:",e )
            for _step in range(self.n_search):
                best_estimate, score = self.step(_step, self.s_word, self.s_image)
                episode_score.append(score)
                episode_image = self.getPerm(self.image_idx, best_estimate)
                if self.paracc:
                    episode_accuracy.append(mapping_accuracy(np.arange(len(self.s_word)), episode_image))# get word idex 
            if self.paracc:
                accuracies.append(episode_accuracy)
            scores.append(episode_score)
            buffer.append(best_estimate)
        
        return self.getPerm(np.arange(self.n_item), np.mean(buffer).astype(int)), scores, accuracies
