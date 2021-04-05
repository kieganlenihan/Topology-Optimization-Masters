from fenics import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy import linalg
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
        A, b = assemble_system(self.a, self.L, self.bc)
        U = self.u.vector()
        solve(A, U, b)
        return np.expand_dims(U.get_local(), axis = 1), A.array()
    def plotSol(self):
        p = plot(self.u)
        plt.colorbar(p)
        plt.show()
def J(kappa):
    T, _ = hSolver.solve(kappa[0])
    return np.linalg.norm(T - truth) ** 2
def J_grad(kappa):
    T, H_k = hSolver.solve(kappa[0])
    H = H_k * kappa[0]
    J_T = T - truth
    w = linalg.solve(H.T, J_T)
    ans = -T.T @ H_k @ w
    return ans
def callbackFunc(x):
    global iter, inter_sol
    iter += 1
    inter_sol.append(x[0])
if __name__ == '__main__':
    print('DEBUG:://\n\n\n\n\n\n\n')
    ## Get ground truth
    n_el_x = 16
    source_val = 10
    k_max = 1E3
    tol = 1E-14
    hSolver = HeatSolverSS(n_el_x, source_val)
    hSolver.constructMesh()
    hSolver.boundaryConditions()
    hSolver.variationalProblem()
    truth, _ = hSolver.solve(5)
    # hSolver.plotSol()
    ## Set search parameters
    x0 = [.0001]
    iter = 0
    inter_sol = []
    ## Optimize
    bnds = Bounds(0 + tol, k_max)
    res = minimize(J, x0, jac = J_grad, method = 'L-BFGS-B', callback = callbackFunc, bounds = bnds, tol = tol)
    print(iter)
    print(inter_sol)