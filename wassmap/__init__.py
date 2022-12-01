# Author: Nick Henscheid <nick.henscheid@gmail.com>
#         Keaton Hamm    <keaton.hamm@uta.edu>
#
# License: MIT License

from .distutils import *
from .imageutils import *

# Interface functions

def wassmap(image_list,d=2,p=2.0,geodesic=False,method='eps',eps=0,k=3,returngraph=False,plot=False):
    """
    Computes the Wassmap embedding for a list of images in pointcloud format
    Options: d is the embedding dimension 
             p is the Wasserstein distance power 
             geodesic = True constructs the (epsilon or kNN) distance graph and 
                             runs all-pair-shortest-path to estimate geodesics 
                             in Wasserstein space
             method = 'eps' if geodesic=True, uses the epsilon neighbor graph construction
             method = 'knn' if geodesic=True, uses the kNN neighbor graph construction 
             eps is the distance threshold (for geodesic=True and method='eps')
             k is the kNN parameter (for geodesic=True and method='knn')
             returngraph = True returns the graph constructed via eps or kNN (for visualization, typically)
    """
    # Step 0: Check the format of image_list to ensure it is a list of pointclouds 
    
    # Step 1: Compute the pairwise Wasserstein distance matrix (or the geodesic wass matrix, if geodesic=True)
    
    D = wass_matrix(image_list,squared=True,p=p,geodesic=geodesic,eps=eps,plot=plot,returngraph=returngraph,method=method,k=k)
    # Step 2: Compute the MDS embedding 
    Z = mds(dist_matrix, num_components=d, squared=False)
    
    return Z
    
def isomap(image_list,p=2.0,geodesic=True,method='eps',eps=0.1,k=3,returngraph=False):
    """
    Computes the Isomap embedding for a list of images in voxel format
    Options:  
             geodesic = True constructs the (epsilon or kNN) distance graph and 
                             runs all-pair-shortest-path to estimate geodesics 
                             in Wasserstein space
             method = 'eps' if geodesic=True, uses the epsilon neighbor graph construction
             method = 'knn' if geodesic=True, uses the kNN neighbor graph construction 
             eps is the distance threshold (for geodesic=True and method='eps')
             k is the kNN parameter (for geodesic=True and method='knn')
    """
    
    # Step 0: check the format of image_list to ensure it is a list of voxel arrays OR an image tensor of size Mx2xN 
    
    # Step 1: Compute the L2 geodesic matrix 
    
    euc_matrix = wm.euc_matrix(trans_images_vox,geodesic=False,squared=True,plot=False)

    # Compute the Euclidean MDS embedding 
    EucEmbedding = wm.mds(euc_matrix,squared=True)
    
    # Step 3: Compute the MDS embedding 
    
    return Z