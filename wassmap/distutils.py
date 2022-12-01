# Distance functions and associated utilities

import ot
import ot.plot
import math
import numpy as np
import numpy.linalg as la
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

# Wasserstein distance and matrix functions 
def uniform_wass_squared(U,V,p=2,Itermax=100000):
    """ 
    Computes the squared Wasserstein distance between two point clouds U and V
    Assumes that U and V are in pointcloud format i.e. U = [x1,y1,w1;...;xp,yp,wp] etc
    """
    Upts = np.ascontiguousarray(U[:,0:2])
    Vpts = np.ascontiguousarray(V[:,0:2])
    Uwts = np.ascontiguousarray(U[:,2])
    Vwts = np.ascontiguousarray(V[:,2])
    Uwts = Uwts/np.sum(Uwts)
    Vwts = Vwts/np.sum(Vwts)
    M = np.power(ot.dist(Upts, Vpts,'euclidean'),p)  # Compute euclidean distance on the pointcloud points
    W = ot.emd2(Uwts,Vwts, M,numItermax=Itermax)
    return W

def wass_matrix(image_list,squared=True,p=2.0,geodesic=False,eps=0,plot=False,returngraph=False,method='eps',k=3):
    """
    The function compute the (squared if squared=True) Wasserstein Distance Matrix between N images
    image_list: python list of pointcloud representations 
    """
    N = len(image_list) #number of images
    wass_dist = np.zeros((N,N)) #initialize the distance matrix
    for i in range(N):
        for j in range(i+1,N):
            wass_dist[i,j] = uniform_wass_squared(image_list[i], image_list[j],p=p)**(1.0/p)
    wass_dist += wass_dist.T

    if(geodesic==True):
        if(plot==True):
            distances   = list(wass_dist[np.triu_indices(N,k=1)])
            print('minimum nonzero distance = %1.4f'%min(distances))
            histfig,histax = plt.subplots()
            histax.hist(distances)
        wass_matrix = np.copy(wass_dist)
        if(returngraph==True):
            wass_dist,G = geodesic_matrix(wass_matrix,method,k,eps,plot,returngraph=True)
        else:
            wass_dist = geodesic_matrix(wass_matrix,method,k,eps,plot)
    if(squared==True):
        wass_dist = np.square(wass_dist) 
    if (geodesic==True&returngraph==True):
        return wass_dist,G;
    else:
        return wass_dist

def euc_matrix(image_tensor,squared=True,geodesic=False,eps=0,plot=False):
    """
    Compute the (squared if squared=True) Euclidean Distance Matrix between N 2D images
    image_tensor: Should be a Mx2xN array, where M is the number of pixels.
    """
    N = image_tensor.shape[-1] #number of images 
    euc_dist = np.zeros((N,N)) #initialize the distance matrix
    for i in range(N):
        for j in range(i+1,N):
            euc_dist[i,j] = la.norm(image_tensor[:,:,i]-image_tensor[:,:,j])
    euc_dist += euc_dist.T

    if(geodesic==True):
        if(plot==True):
            distances   = list(euc_dist[np.triu_indices(N,k=1)])
            print('minimum nonzero distance = %1.4f'%min(distances))
            histfig,histax = plt.subplots()
            histax.hist(distances)
        euc_matrix = np.copy(euc_dist)
        euc_dist = geodesic_matrix(euc_matrix,eps,plot)

    if(squared==True):
        euc_dist = np.square(euc_dist) 

    return euc_dist

def geodesic_matrix(dist_matrix,method='eps',k=3,eps=1,plot=False,returngraph=False):
    """
    Computes the Geodesic distance matrix given a pairwise distance matrix 
    threshold eps 
    """
    bignumber = 1e2
    N = dist_matrix.shape[0]

    if method=='eps':
        dist_matrix[distance_matrix>eps] = 0
        G = nx.from_numpy_array(dist_matrix)
        if(plot==True):
            graphfig,graphax = plt.subplots()
            nx.draw_networkx(G,pos=nx.spring_layout(G),ax=graphax)
    elif method=='knn':
        A = kneighbors_graph(X=dist_matrix,n_neighbors=k,metric='precomputed', mode='distance', include_self=False)
        G = nx.from_numpy_array(A.toarray())
        if(plot==True):
            graphfig,graphax = plt.subplots()
            nx.draw_networkx(G,pos=nx.spring_layout(G),ax=graphax)
    else: 
        print('ERROR: Invalid method given! method=\'eps\' or method=\'knn\'')

    D = dict(nx.all_pairs_dijkstra_path_length(G))
    #display(D)
    dist = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            try:
                dist[i,j] = D[i][j]
            except KeyError:
                # couldn't find path from i to j, so set distance to infinite
                dist[i,j] = bignumber
    dist += dist.T
    if(returngraph==True):
        return dist,G;
    else:
        return dist

def mds(dist_matrix, num_components=2, squared=False):
    """
    Computes non-metric (classical) Multidimensional Scaling (MDS)
    Input should be an NxN distance matrix
    The Boolean squared indicates if the distance matrix passed to the function has already
    been squared entrywise (True) or not (False)
    """
    N = dist_matrix.shape[0]
    H = np.eye(N)-1/N*np.ones((N,N))
    if squared==False:
        B = -.5*H@(dist_matrix**2)@H
    else:
        B = -.5*H@dist_matrix@H
    U,S,VT = la.svd(B)
    embedding = U[:,:num_components]@np.diag(S[:num_components]**.5)
    return embedding