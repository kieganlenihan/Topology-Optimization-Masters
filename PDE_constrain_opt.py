from fenics import *
import numpy as np
from PDE_Solver import *

class HeatSolver(PDE_Solver):
    def __init__(self):
        pass
    def constructMesh(self):
        self.mesh = mesh
    def constructFunctionSpace(self):
        # set Elements
        self.V = FunctionSpace(self.mesh, element_type)
    def setRobinBCs(self):
        pass
    def setNeumanBCs(self):
        pass
    def setDirichletBCs(self):
        pass
    
