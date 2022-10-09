# Functions for voxel reps and pointcloud reps
# NOTE: classical Isomap is defined for voxelized images, not pointclouds.
import ot
import ot.plot
import math
import numpy as np
import numpy.linalg as la
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

def vox_to_pointcloud(voxarray,grid,eps=0):
    # Convert a voxel representation ('voxarray') to a (weighted) point cloud representation 
    # Assume that the input grid is in "ij meshgrid" format i.e. the grid has two layers, xx and yy, each size (nx)-by-(ny)
    # The voxel array will be unrolled via "column-major" order ("Fortran/Matlab" ordering)
    # Note that initially the number of voxels must equal the number of grid points - but 
    # zero voxels will be removed from the representation (i.e. no points with weight zero allowed)
    # if the optional parameter eps is passed, voxels with value less or equal to eps will be dropped
    # The returned array consists of (x,y,w) tuples i.e. X = [x1,y1,w1;x2,y2,w2;...;xP,yP,wP] where P 
    # is the number of nonzero points 
    xx,yy = np.squeeze(np.split(grid,2))
    X = np.vstack((xx.ravel(),yy.ravel())).T
    nX  = X.shape[0] # Number of points = number of rows
    nvi = voxarray.shape[0] # Number of voxel rows
    nvj = voxarray.shape[1] # Number of voxel cols 
    if nX != nvi*nvj: raise ValueError("Number of grid points must equal number of voxels!")
    X = np.concatenate((X,voxarray.T.reshape(nvi*nvj,1)),axis=1)
    return X[X[:,2]>eps,:]

def pointcloud_to_vox(array,grid):
    # Converts a pointcloud representation to a voxel representation 
    # Assumes that the grid is in "ij meshgrid" format i.e. grid has two layers, xx and yy; each are size (nx+1)-by-(ny+1)
    # The grid points are assumed to define the corners of the voxels, so the 
    # voxel rep will be a single nx-by-ny array with entries equal to the average pointcloud weights
    # V_ij = \mean_k W_k if (x_k,y_k) is in voxel ij
    xx,yy = np.squeeze(np.split(grid,2))
    points = array[:,0:2]
    values = array[:,2]
    interp = LinearNDInterpolator(points,values,fill_value=0.0)
    X = interp(xx,yy).T
    return X

# Synthetic image generation functions
def generate_rectangle(side0, side1, initial_point=[0,0], samples=100):
    # Generates a rectangle in point cloud format
    x = np.linspace(initial_point[0], initial_point[0]+side0, num=samples)
    y = np.linspace(initial_point[1], initial_point[1]+side1, num=samples)
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

def in_triangle(endpoint1,endpoint2,endpoint3,point):
    # Indicator function of a triangle 
    # Returns 1 (True) if point is in the triangle, zero (False) else
    c1 = (endpoint2[0]-endpoint1[0])*(point[1]-endpoint1[1]) - (endpoint2[1]-endpoint1[1])*(point[0]-endpoint1[0])
    c2 = (endpoint3[0]-endpoint2[0])*(point[1]-endpoint2[1]) - (endpoint3[1]-endpoint2[1])*(point[0]-endpoint2[0])
    c3 = (endpoint1[0]-endpoint3[0])*(point[1]-endpoint3[1]) - (endpoint1[1]-endpoint3[1])*(point[0]-endpoint3[0])

    if (c1<0 and c2<0 and c3<0) or (c1>0 and c2>0 and c3>0):
        return True
    else:
        return False    

def generate_triangle(endpoint1, endpoint2, endpoint3, samples=100):
    # Generates a triangle in point cloud format
    x = np.linspace(min(endpoint1[0],endpoint2[0],endpoint3[0]), max(endpoint1[0],endpoint2[0],endpoint3[0]), num=samples)
    y = np.linspace(min(endpoint1[1],endpoint2[1],endpoint3[1]), max(endpoint1[1],endpoint2[1],endpoint3[1]), num=samples)
    xy_0 = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    xy = []
    for point in xy_0:
        if in_triangle(endpoint1,endpoint2,endpoint3,point):
            xy.append(point)
    return np.array(xy)

# Image transformation functions
def rotation(object, radian_degree):
    A = [[math.cos(radian_degree), -math.sin(radian_degree)],[math.sin(radian_degree), math.cos(radian_degree)]]
    image = []
    for index,point in enumerate(object):
        image.append(np.matmul(A,point))
    return np.array(image)
def translation(object, translate_direction):
    object_array = np.array(object)
    direction_array = np.array(translate_direction)
    image = [x + direction_array for x in object_array]
    return np.array(image)

def dilation(object, parameter):
    A = [[parameter[0], 0],[0, parameter[1]]]
    image = []
    for index,point in enumerate(object):
        image.append(np.matmul(A,point))
    return np.array(image)

def in_circle(center, radius, point):
    if (point[1]-center[1])**2+(point[0]-center[0])**2<=radius**2:
        return True
    else:
        return False

def generate_circle(center, radius, samples=100):
    x = np.linspace(center[0]-radius, center[0]+radius, num=samples)
    y = np.linspace(center[1]-radius, center[1]+radius, num=samples)
    xy_0 = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    xy = []
    for point in xy_0:
        if in_circle(center,radius,point):
            xy.append(point)
    return np.array(xy)

def generate_ellipse(center, axis_x, axis_y,samples = 100):
    circle = generate_circle([0,0],1,samples)
    ellipse = dilation(circle,[axis_x, axis_y])
    ellipse = translation(ellipse,center)
    return np.array(ellipse)

# Distance matrix computation functions
def uniform_Wass_squared(U, V,p=2,Itermax=100000):
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

def Wasserstein_Matrix(image_list,squared=True,p=2.0,geodesic=False,eps=0,plot=False,returngraph=False,method='eps',k=3):
    """
    The function compute the (squared if squared=True) Wasserstein Distance Matrix between N images
    image_list: python list of pointcloud representations 
    """
    N = len(image_list) #number of images
    wass_distance = np.zeros((N,N)) #initialize the distance matrix
    for i in range(N):
        for j in range(i+1,N):
            wass_distance[i,j] = uniform_Wass_squared(image_list[i], image_list[j],p=p)**(1.0/p)
    wass_distance += wass_distance.T

    if(geodesic==True):
        if(plot==True):
            distances   = list(wass_distance[np.triu_indices(N,k=1)])
            print('minimum nonzero distance = %1.4f'%min(distances))
            histfig,histax = plt.subplots()
            histax.hist(distances)
        wass_matrix = np.copy(wass_distance)
        if(returngraph==True):
            wass_distance,G = Geodesic_Matrix(wass_matrix,method,k,eps,plot,returngraph=True)
        else:
            wass_distance = Geodesic_Matrix(wass_matrix,method,k,eps,plot)

    if(squared==True):
        wass_distance = np.square(wass_distance) 
    if (geodesic==True&returngraph==True):
        return wass_distance,G;
    else:
        return wass_distance

def Euclidean_Matrix(image_tensor,squared=True,geodesic=False,eps=0,plot=False):
    """
    Compute the (squared if squared=True) Euclidean Distance Matrix between N 2D images
    image_tensor: Should be a Mx2xN array, where M is the number of pixels.
    """
    N = image_tensor.shape[-1] #number of images 
    euc_distance = np.zeros((N,N)) #initialize the distance matrix
    for i in range(N):
        for j in range(i+1,N):
            euc_distance[i,j] = la.norm(image_tensor[:,:,i]-image_tensor[:,:,j])
    euc_distance += euc_distance.T

    if(geodesic==True):
        if(plot==True):
            distances   = list(euc_distance[np.triu_indices(N,k=1)])
            print('minimum nonzero distance = %1.4f'%min(distances))
            histfig,histax = plt.subplots()
            histax.hist(distances)
        euc_matrix = np.copy(euc_distance)
        euc_distance = Geodesic_Matrix(euc_matrix,eps,plot)

    if(squared==True):
        euc_distance = np.square(euc_distance) 

    return euc_distance

def Geodesic_Matrix(distance_matrix,method='eps',k=3,eps=1,plot=False,returngraph=False):
    """
    Computes the Geodesic distance matrix given a pairwise distance matrix 
    threshold eps 
    """
    bignumber = 1e2
    N = distance_matrix.shape[0]

    if method=='eps':
        distance_matrix[distance_matrix>eps] = 0
        G = nx.from_numpy_array(distance_matrix)
        if(plot==True):
            graphfig,graphax = plt.subplots()
            nx.draw_networkx(G,pos=nx.spring_layout(G),ax=graphax)
    elif method=='knn':
        A = kneighbors_graph(X=distance_matrix,n_neighbors=k,metric='precomputed', mode='distance', include_self=False)
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

def MDS(distance_matrix, num_components=2, squared=False):
    """
    Computes non-metric (classical) Multidimensional Scaling (MDS)
    Input should be an NxN distance matrix
    The Boolean squared indicates if the distance matrix passed to the function has already
    been squared entrywise (True) or not (False)
    """
    N = distance_matrix.shape[0]
    H = np.eye(N)-1/N*np.ones((N,N))
    if squared==False:
        B = -.5*H@(distance_matrix**2)@H
    else:
        B = -.5*H@distance_matrix@H
    U,S,VT = la.svd(B)
    embedding = U[:,:num_components]@np.diag(S[:num_components]**.5)
    return embedding