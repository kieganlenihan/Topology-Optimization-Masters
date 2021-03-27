from fenics import *

class PDE_solver:
    def __init__(self):
        pass
    def defineObjectiveFunction(self):
        pass
    def defReducedObjectiveFunction(self, J, m):
        self.J_hat = ReducedFunctional(J, m)
    def assembleSystem(self):
        pass
    def assembleRHS(self):
        pass
    def solve(self):
        pass
    def getSolution(self):
        pass
    def getNumNodes(self):
        pass