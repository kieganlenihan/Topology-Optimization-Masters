from fenics import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
import numpy as np

class HeatSolverSS():
    def __init__(self, side, source):
        self.side = side
        self.source = source
    def constructMesh(self):
        self.mesh = UnitSquareMesh(self.side, self.side)
        self.V = FunctionSpace(self.mesh, 'P', 1)
    def boundaryConditions(self):
        u_D = Constant(0.0)
        def boundary(x, on_boundary):
            return on_boundary and near(x[1], 0, tol)
        self.bc = DirichletBC(self.V, u_D, boundary)
        boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 1)
        class source(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 1, tol)
        bx = source()
        bx.mark(boundary_markers, 0)
        self.ds = Measure("ds", domain = self.mesh, subdomain_data = boundary_markers)
    def variationalProblem(self):
        u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.a = dot(grad(u), grad(self.v)) * dx
        self.q = Constant(self.source)
        self.a = dot(grad(u), grad(self.v)) * dx
    def solve(self, k):
        self.L = 1/k *self.q * self.v * self.ds(0)
        self.u = Function(self.V)
        solve(self.a == self.L, self.u, self.bc)
        return self.u.vector().get_local()
    def plotSol(self):
        p = plot(self.u)
        plt.colorbar(p)
        plt.show()
def J(kappa):
        return np.linalg.norm(hSolver.solve(kappa[0]) - truth) ** 2
def callbackFunc(x):
    global iter, inter_sol
    iter += 1
    inter_sol.append(x[0])
if __name__ == '__main__':
    ## Get ground truth
    n_el_x = 16
    source_val = 10
    k_max = 1E3
    tol = 1E-14
    hSolver = HeatSolverSS(n_el_x, source_val)
    hSolver.constructMesh()
    hSolver.boundaryConditions()
    hSolver.variationalProblem()
    truth = hSolver.solve(5)
    # hSolver.plotSol()
    ## Set search parameters
    kappa = 1
    x0 = [-1.0]
    iter = 0
    pk = -1.0
    inter_sol = []
    inter_sol.append(kappa)
    ## Optimize
    bnds = Bounds(0 + tol, k_max)
    res = minimize(J, x0, method = 'L-BFGS-B', callback = callbackFunc, bounds = bnds, tol = tol)
    print(res.x)
    print(inter_sol)