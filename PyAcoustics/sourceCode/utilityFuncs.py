from fenics import *
import numpy as np
from scipy import linalg

def findNodesInsideDomain(meshManager, targetSubdomain):
    all_coordinates = meshManager.mesh.coordinates()
    mesh1 = SubMesh(meshManager.mesh, meshManager.mf_domain,targetSubdomain)
    coordinates=mesh1.coordinates()

    X = np.array([[np.min(coordinates[:,0], 0), np.max(coordinates[:,0],0)]])
    Y = np.array([[np.min(coordinates[:,1], 0), np.max(coordinates[:,1],0)]])
    Z = np.array([[np.min(coordinates[:,2], 0), np.max(coordinates[:,2],0)]])

    found_indexes = []
    for i, row in enumerate(all_coordinates):
        if X[0,0] <= row[0] and row[0] <= X[0,1] and Y[0,0] <= row[1] and row[1] <= Y[0,1] and Z[0,0] <= row[2] and row[2] <= Z[0,1]:
            found_indexes.append(i)
    return found_indexes

def findIndexOfCenterNode(mesh):
    coordinates = mesh.coordinates()
    X = np.array([[np.min(coordinates[:,0], 0), np.max(coordinates[:,0],0)]])
    Y = np.array([[np.min(coordinates[:,1], 0), np.max(coordinates[:,1],0)]])
    Z = np.array([[np.min(coordinates[:,2], 0), np.max(coordinates[:,2],0)]])
    x_c=(X[0,0]+X[0,1])/2
    y_c=(Y[0,0]+Y[0,1])/2
    z_c=(Z[0,0]+Z[0,1])/2
    center_point = np.array([[x_c, y_c, z_c]])
    
    n = len(coordinates)
    distance=np.zeros((n,1))
   
    for i, row in enumerate(coordinates):
        distance[i] = linalg.norm(row.T - center_point)
  
    index = np.nonzero(distance == np.min(distance))
    return index[0]

    
    





