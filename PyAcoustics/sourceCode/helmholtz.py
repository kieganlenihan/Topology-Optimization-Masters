import sys
sys.path.insert(1, '/Users/kieganlenihan/Documents/Fenics_Masters/PyExoMeshConverter')
from MeshReader import *
from fenics import *
import numpy as np
from PDE_Solver import *
from utilityFuncs import *
import copy

class AcousticSolverComplex(PDE_Solver):
    def __init__(self, c, f, meshPrefix, p_order, linearsolver, xmdf_mesh_exists):
        PDE_Solver.__init__(self)
        #self.boundaryConditions = boundaryConditions
        self.xmdf_mesh_exists=xmdf_mesh_exists
        self.meshPrefix = meshPrefix
        self.constructMesh()
        self.c = c
        self.frequency =f
        self.p_order = p_order
        self.linearsolver=linearsolver
        self.constructFunctionSpace()
        self.K = []
        self.M = []
        self.C = []
        self.F = []
        self.TransferMat = []
        self.solver=[]
        print("Exodus File: {:s}".format(meshPrefix))
    def constructMesh(self):
        self.meshManager = MeshReader(self.meshPrefix, not self.xmdf_mesh_exists)
        self.mesh=self.meshManager.mesh
        self.dx = self.meshManager.dx
        self.ds = self.meshManager.ds

    def constructFunctionSpace(self):
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.p_order)
        TH = P1 * P1
        self.V = FunctionSpace(self.mesh, TH)
        self.pressure =Function(self.V)
    def setPointSources(self, pointSources):
        self.pointSources = pointSources

    def setImpedanceBCs(self, impedanceBCs):
        self.impedanceBCs = impedanceBCs
    def setNaturalBCs(self, naturalBCs):
        self.naturalBCs = naturalBCs
    def addNaturalBCs(self):
        if not self.naturalBCs:
            return
        pr, pi = TrialFunction(self.V)
        vr, vi = TestFunction(self.V)

        integrals_Fn = []
        for bc in self.naturalBCs:
            accel = bc['acceleration']
            accel_real_c = Constant(-accel.real)
            accel_imag_c = Constant(-accel.imag)
            sidesets = bc['sidesets']
            n = len(sidesets)
            assert(n == 1), 'We do not support more than one sideset per natural bc so far.'
            for q in sidesets:
                integrals_Fn.append((accel_real_c * vr + accel_imag_c * vi) * self.ds(q))

            self.F = assemble(sum(integrals_Fn))


    def assembleImpedanceMatrix(self):
        if not self.impedanceBCs:
            return
        pr, pi = TrialFunction(self.V)
        vr, vi = TestFunction(self.V)

        integrals_C = []
        for bc in self.impedanceBCs:
            Z = bc['impedance']
            sidesets = bc['sidesets']
            for q in sidesets:
                integrals_C.append( 1/Z * (pr * vi - pi * vr) * self.ds(q))

        self.C = assemble(sum(integrals_C))

    def addPointSourcesToForceVector(self):
        if not self.F:
            f = Constant(0.0)
            vr, vi = TestFunction(self.V)
            L = f * (vr + vi) * dx
            self.F = assemble(L)

        for source in self.pointSources:
           magnitude = source['amplitude'].real
           delta = PointSource(self.V.sub(0), source['location'], magnitude)
           delta.apply(self.F)
           magnitude_imag = source['amplitude'].imag
           delta_imag = PointSource(self.V.sub(1), source['location'], magnitude_imag)
           delta_imag.apply(self.F)

    def assembleSystemMatrices(self):
        pr, pi = TrialFunction(self.V)
        vr, vi = TestFunction(self.V)
        K = dot(grad(pr), grad(vr)) * self.dx  + dot(grad(pi), grad(vi)) * self.dx
        M = pr * vr * self.dx + pi * vi * self.dx
        self.M = assemble(M)
        self.K=assemble(K)
        self.assembleImpedanceMatrix()

    def assembleRHS(self):
        self.addNaturalBCs()
        self.addPointSourcesToForceVector()

    def defineDirichletConditions(self):
        pass
        # def boundary(x, on_boundary):
        #     return on_boundary
        # self.dbc = DirichletBC(self.V, Constant(0), boundary)
    def assembleSystem(self):
        self.assembleSystemMatrices()
        self.assembleRHS()
        self.defineDirichletConditions()
    def solve(self):
        if not self.K or not self.F:
            self.assembleSystem()

        if not self.solver:
            omega = 2*pi*self.frequency
            kappa = omega/self.c
            self.A = self.K - kappa**2 * self.M + omega * self.C
            self.solver = LUSolver(self.A,'mumps')

        #self.dbc.apply(A,self.F)
        #Using a LUSolver object is much faster as PetSC reuses the factorization.
        self.solver.solve(self.pressure.vector(),self.F)
        #solve(self.A,self.pressure.vector(),self.F, self.linearsolver)

    def getSolution(self):
        return self.pressure
    def getNumNodes(self):
        return self.mesh.num_vertices()

    def outputSolution(self, prefix):
        vtkfile = File('./results/'+prefix+'_pressure.pvd')
        p_real, p_imag = self.pressure.split(True)
        vtkfile << p_real
        vtkfile << p_imag
    def getTransferMatrix(self, fileprefix,  targetSubdomain):
        found_indexes = findNodesInsideDomain(self.meshManager,targetSubdomain)

        temp_NBCs = copy.deepcopy(self.naturalBCs)

        #Construct Transfer matrix for Natural boundary conditions
        numNodes = len(found_indexes)
        numBcs  = len(self.naturalBCs)
        self.TransferMat=np.zeros((numNodes,numBcs), dtype=np.cfloat)

        for i, nbc in enumerate(self.naturalBCs):
            print("Generating column:", i)
            nbc["acceleration"]=complex(1+0j)
            self.setNaturalBCs([nbc])
            self.assembleRHS()
            self.solve()
            sol = self.getSolution()
            sol_real, sol_im = sol.split()
            # Store the information using node ordering as in the mesh coordinates
            self.TransferMat[:,i] = sol_real.compute_vertex_values()[found_indexes]+ sol_im.compute_vertex_values()[found_indexes]*1j

        if type(fileprefix) is str:
            np.save("./npyFiles/"+fileprefix+"_TransferMatrix", self.TransferMat)
        #put back the original BCs
        self.naturalBCs = copy.deepcopy(temp_NBCs.copy)
        return self.TransferMat

    def loadTransferMatrix(self, fileprefix):
        assert fileprefix, "No file specified for loading transfer matrix"
        self.TransferMat = np.load("./npyFiles/"+fileprefix+"_TransferMatrix"+".npy")
        return self.TransferMat
    def solveForPSD(self, sourcePSD, nodeArray=np.array([0])):
        if nodeArray.any():
            reducedTransferMat =  self.TransferMat[nodeArray,:]
            sol = reducedTransferMat.dot(sourcePSD).dot(reducedTransferMat.conj().T)
            return sol
        else:
            return self.TransferMat.dot(sourcePSD).dot(self.TransferMat.conj().T)
    def solveForPSDRow(self, sourcePSD, row):
            transferMatRow =  self.TransferMat[row,:]
            sol = transferMatRow.dot(sourcePSD).dot(self.TransferMat.conj().T)
            return sol
