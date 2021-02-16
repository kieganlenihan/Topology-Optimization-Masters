from fenics import *
from helmholtz import *
import numpy as np
import sys
import math
sys.path.insert(1, '/Users/kieganlenihan/Documents/Fenics_Masters/PyExoMeshConverter')
from MeshReader import *

#Miscellaneous inputs
freq=400
omega = 2*math.pi*freq
c=340
Z=2.25e4
kappa = omega/c
p_order = 1
xmdf_mesh_exists = False
target_subdomain = 1 # subdomain for transfer function
##### 3D #############################################
speakers = 36
case="forwardSolution_test"
exodus_file="room634_36spkr_0.1"
#********************FE Solver Parameters
#Impedance BCs
impendanceBcs =[{"sidesets": [1], "impedance": Z}]

#Neumann BCs
natural_bcs = []
for s in range(2,speakers+2):
    dict = {"sidesets":[s], "acceleration":1+0j}
    natural_bcs.append(dict)
pointSources = []

Asolver = AcousticSolverComplex(c,freq, exodus_file, p_order, 'mumps', xmdf_mesh_exists)
Asolver.setPointSources(pointSources)
Asolver.setImpedanceBCs(impendanceBcs)
Asolver.setNaturalBCs(natural_bcs)
Asolver.solve()
Asolver.outputSolution(case)
