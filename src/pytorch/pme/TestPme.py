import torch
import numpy as np
import pytest
from NNPOps.pme import PME

from openmm import *
system = System()
nb = NonbondedForce()
nb.setNonbondedMethod(NonbondedForce.PME)
nb.setCutoffDistance(0.5)
nb.setEwaldErrorTolerance(1e-3)
nb.setReciprocalSpaceForceGroup(1)
system.addForce(nb)
for i in range(9):
    system.addParticle(1.0)
    nb.addParticle((i-4)*0.1, 0, 0)
system.setDefaultPeriodicBoxVectors(Vec3(1, 0, 0), Vec3(0, 1.1, 0), Vec3(0, 0, 1.2))
integrator = VerletIntegrator(1.0)
context = Context(system, integrator)
np.random.seed(10)
pos = np.random.rand(9*3).reshape((9, 3))
context.setPositions(pos)
print(nb.getPMEParametersInContext(context))
print(context.getState(getEnergy=True, groups={0}).getPotentialEnergy())
print(context.getState(getEnergy=True, groups={1}).getPotentialEnergy())

pme = PME(14, 15, 16, 5, 4.985823141035867, 138.935)
positions = torch.tensor(pos, dtype=torch.float32)
charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32)
box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32)
print(pme.compute_direct(positions, charges, 0.5, box_vectors))
print(pme.compute_reciprocal(positions, charges, box_vectors))