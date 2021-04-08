from fenics import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy import linalg
import numpy as np

class source(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1, tol)
class HeatSolverSS():
    def __init__(self, side, source):
        self.side = side
        self.source = source
    def constructMesh(self):
        self.mesh = UnitSquareMesh(self.side, self.side)
        self.V = FunctionSpace(self.mesh, 'CG', 1)
    def boundaryConditions(self):
        u_D = Constant(0.0)
        def boundary(x, on_boundary):
            return on_boundary and near(x[1], 0, tol)
        self.bc = DirichletBC(self.V, u_D, boundary)
        boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 1)
        bx = source()
        bx.mark(boundary_markers, 0)
        self.ds = Measure("ds", domain = self.mesh, subdomain_data = boundary_markers)
    def initial_guess(self, guess):
        self.x = interpolate(guess, self.V)
        return np.array(self.x.vector())
    def variationalProblem(self):
        self.U = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.q = Constant(self.source)
        self.B = Function(self.V)
    def solve(self, k):
        self.B.vector()[:] = k
        kappa = interpolate(self.B, self.V)
        self.a = dot(grad(self.U), kappa * grad(self.v)) * dx
        self.L = self.q * self.v * self.ds(0)
        self.u = Function(self.V)
        A, b = assemble_system(self.a, self.L, self.bc)
        A_ = assemble(dot(grad(self.U), grad(self.v)) * dx)
        U = self.u.vector()
        solve(A, U, b)
        return np.expand_dims(U.get_local(), axis = 1), A.array()
    def plotSol(self):
        p = plot(self.u)
        plt.colorbar(p)
        plt.show()
def J(kappa):
    global T, H
    T, H = hSolver.solve(kappa)
    return np.linalg.norm(T - truth) ** 2
def J_grad(kappa):
    J_T = T - truth
    w = linalg.solve(H, J_T)
    ans = -H @ w
    return ans
def callbackFunc(x):
    global iter, inter_sol
    iter += 1
    inter_sol.append(x)
if __name__ == '__main__':
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
    ## Get ground truth
    n_el_x = 16
    source_val = 10
    k_max = 1E3
    tol = 1E-14
    true_val = 1.4
    hSolver = HeatSolverSS(n_el_x, source_val)
    hSolver.constructMesh()
    hSolver.boundaryConditions()
    hSolver.variationalProblem()
    truth, _ = hSolver.solve(Constant(true_val))
    # hSolver.plotSol()
    ## Set search parameters
    iter = 0
    x0 = 1.5
    x0 = hSolver.initial_guess(Constant(x0))
    inter_sol = []
    ## Optimize
    # bnds = Bounds(0 + tol, k_max)
    res = minimize(J, x0, jac = J_grad, method = 'L-BFGS-B', callback = callbackFunc, tol = tol)
    # hSolver.plotSol()
    print('Iterations to complete', iter)
    # print(inter_sol)