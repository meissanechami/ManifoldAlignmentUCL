from data_handler import *
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import cdist,pdist, squareform
from utils import now
from MAGAN.model import MAGAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import Isomap, SpectralEmbedding
from MAGAN.loader import Loader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.cm
from scipy.stats import spearmanr

def correspondence_loss(b1, b2):
    """
    The correspondence loss.

    :param b1: a tensor representing the object in the graph of the current minibatch from domain one
    :param b2: a tensor representing the object in the graph of the current minibatch from domain two
    :returns a scalar tensor of the correspondence loss
    """
    domain1cols = [x for x in range(b1.shape[1])]
    domain2cols = [x for x in range(b2.shape[1])]
    loss = tf.constant(0.)
    for c1, c2 in zip(domain1cols, domain2cols):
        loss += tf.reduce_mean((b1[:, c1] - b2[:, c2])**2)

    return loss

if __name__ == "__main__":

    %matplotlib inline
    results = []
    xb1_train, xb2_train, labels1_train, labels2_train, xb1_test, xb2_test, labels1_test, labels2_test= get_data("gauss", ssl=True)

    # Prepare the loaders
    loadb1 = Loader(xb1_train, labels=labels1_train, shuffle=False)
    loadb2 = Loader(xb2_train, labels=labels2_train, shuffle=False)

    batch_size = 50

    tf.reset_default_graph()
    # Build the tf graph
    magan = MAGAN(dim_b1=[None, xb1_train.shape[1]], dim_b2=[None, xb2_train.shape[1]], correspondence_loss=correspondence_loss, learning_rate=0.001)

    # Train
    for i in range(1, 10000):
        if i % 1000 == 0: print("Iter {} ({})".format(i, now()))
        xb1_, label1 = loadb1.next_batch(batch_size)
        xb2_, label2 = loadb2.next_batch(batch_size)

        magan.train(xb1_, xb2_)

          # Evaluate the loss and plot
          if i % 100 == 0:
              xb1_, label1 = loadb1.next_batch(10 * batch_size)
              xb2_, label2 = loadb2.next_batch(10 * batch_size)

              lstring = magan.get_loss(xb1_, xb2_)
              ld, lc = lstring.strip().split()
              print("{} {}".format(magan.get_loss_names(), lstring))

              xb1 = magan.get_layer(xb1_, xb2_, 'xb1')
              xb2 = magan.get_layer(xb1_, xb2_, 'xb2')
              Gb1 = magan.get_layer(xb1_, xb2_, 'Gb1')
              Gb2 = magan.get_layer(xb1_, xb2_, 'Gb2')

    xb1_tte = magan.get_layer(xb1_test, xb2_test, 'xb1')
    xb2_tte = magan.get_layer(xb1_test, xb2_test, 'xb2')
    Gb1_tte = magan.get_layer(xb1_test, xb2_test, 'Gb1')
    Gb2_tte = magan.get_layer(xb1_test, xb2_test, 'Gb2')


    xb1_tr = magan.get_layer(xb1_train, xb2_train, 'xb1')
    xb2_tr = magan.get_layer(xb1_train, xb2_train, 'xb2')
    Gb1_tr = magan.get_layer(xb1_train, xb2_train, 'Gb1')
    Gb2_tr = magan.get_layer(xb1_train, xb2_train, 'Gb2')

    acc = 0
    for i in range(len(Gb1_tr)):
      distances = cdist(xb2_tr[i].reshape(1,xb2_tr.shape[1]), Gb1_tr)
      #print(labels1_train[i], labels2_train[np.argmin(distances, axis=1)])
      if labels1_train[i] == labels2_train[np.argmin(distances, axis=1)]:
        acc +=1
    train_acc = acc/len(Gb1_tr)

    acc = 0
    for i in range(len(Gb1_tte)):
      distances = cdist(xb2_tte[i].reshape(1,xb2_tte.shape[1]), Gb1_tte)
      #print(labels1_train[i], labels2_train[np.argmin(distances, axis=1)])
      if labels1_test[i] == labels2_test[np.argmin(distances, axis=1)]:
        acc +=1
    test_acc=acc/len(Gb1_tte)