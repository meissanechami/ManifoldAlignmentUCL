import numpy as np

def Procrustes(X,Y):
    M=min(X.shape[1], Y.shape[1])

    #mean
    X_mean= np.mean(X.T, axis=0)
    Y_mean= np.mean(Y.T, axis=0)
    
    for i in range(M):
        X[:,i]=X[:,i]-X_mean.T
        Y[:,i]=Y[:,i]-Y_mean.T


    #Procrustes alignment
    u, s, v=np.linalg.svd(Y@X.T)
    v = v.T
    Q=u@v.T
    s = np.expand_dims(s,axis=1)

    k=np.trace(s)/(np.trace(Y@Y.T))

    return Q, k, X_mean, Y_mean