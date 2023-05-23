import torch
import numpy as np
from NNPOps.pme import PME

def test_PME():
    """Compare forces and energies to values computed with OpenMM."""
    pme = PME(14, 15, 16, 5, 4.985823141035867, 138.935)
    pos = [[0.7713206433, 0.02075194936, 0.6336482349],
           [0.7488038825, 0.4985070123, 0.2247966455],
           [0.1980628648, 0.7605307122, 0.1691108366],
           [0.08833981417, 0.6853598184, 0.9533933462],
           [0.003948266328, 0.5121922634, 0.8126209617],
           [0.6125260668, 0.7217553174, 0.2918760682],
           [0.9177741225, 0.7145757834, 0.542544368],
           [0.1421700476, 0.3733407601, 0.6741336151],
           [0.4418331744, 0.4340139933, 0.6177669785]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32)
    edirect = pme.compute_direct(positions, charges, 0.5, box_vectors)
    assert np.allclose(0.5811535194516182, edirect.detach().numpy())
    erecip = pme.compute_reciprocal(positions, charges, box_vectors)
    assert np.allclose(-90.92361028496651, erecip.detach().numpy())
    expected_ddirect = [[-0.4068958163, 1.128490567, 0.2531163692],
                        [8.175477028, -15.20702648, -5.499810219],
                        [-0.2548360825, 0.003096142784, -0.67370224],
                        [0.09854402393, 0.5804504156, 1.063418627],
                        [-0, -0, -0],
                        [-7.859698296, 14.16478539, 5.236941814],
                        [0.684042871, -1.312145352, 0.7057141662],
                        [30.47141075, 6.726415634, -6.697656631],
                        [-30.90804291, -6.084065914, 5.611977577]]
    expected_drecip = [[-0.6407046318, -27.59628105, -3.745499372],
                       [30.76446915, -27.10591507, -82.14082336],
                       [-15.06353951, 10.37030602, -38.38755035],
                       [-7.421859741, 21.9861393, 39.86354828],
                       [-0, -0, -0],
                       [-13.09759808, 6.393665314, 34.15939713],
                       [19.53832817, -59.55260849, 33.96843338],
                       [122.5542908, 60.35510254, -27.44270515],
                       [-136.679245, 15.14429855, 43.89074326]]
    edirect.backward()
    assert np.allclose(expected_ddirect, positions.grad, rtol=1e-4)
    positions.grad.zero_()
    erecip.backward()
    assert np.allclose(expected_drecip, positions.grad, rtol=1e-4)

def test_charge_deriv():
    """Test derivatives with respect to charge."""
    pme = PME(14, 15, 16, 5, 4.985823141035867, 138.935)
    pos = [[0.7713206433, 0.02075194936, 0.6336482349],
           [0.7488038825, 0.4985070123, 0.2247966455],
           [0.1980628648, 0.7605307122, 0.1691108366],
           [0.08833981417, 0.6853598184, 0.9533933462],
           [0.003948266328, 0.5121922634, 0.8126209617],
           [0.6125260668, 0.7217553174, 0.2918760682],
           [0.9177741225, 0.7145757834, 0.542544368],
           [0.1421700476, 0.3733407601, 0.6741336151],
           [0.4418331744, 0.4340139933, 0.6177669785]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, requires_grad=True)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32)

    # Compute derivatives of the energies with respect to charges.

    edir = pme.compute_direct(positions, charges, 0.5, box_vectors)
    erecip = pme.compute_reciprocal(positions, charges, box_vectors)
    edir.backward(retain_graph=True)
    ddir = charges.grad.clone().detach().numpy()
    charges.grad.zero_()
    erecip.backward(retain_graph=True)
    drecip = charges.grad.clone().detach().numpy()

    # Compute finite difference approximations from two displaced inputs.

    delta = 0.001
    for i in range(len(charges)):
        c1 = charges.clone()
        c1[i] += delta
        edir1 = pme.compute_direct(positions, c1, 0.5, box_vectors).detach().numpy()
        erecip1 = pme.compute_reciprocal(positions, c1, box_vectors).detach().numpy()
        c2 = charges.clone()
        c2[i] -= delta
        edir2 = pme.compute_direct(positions, c2, 0.5, box_vectors).detach().numpy()
        erecip2 = pme.compute_reciprocal(positions, c2, box_vectors).detach().numpy()
        assert np.allclose(ddir[i], (edir1-edir2)/(2*delta), rtol=1e-3)
        assert np.allclose(drecip[i], (erecip1-erecip2)/(2*delta), rtol=1e-3)

    # Make sure the chain rule is applied properly.

    charges.grad.zero_()
    (2.5*edir).backward()
    ddir2 = charges.grad.clone().detach().numpy()
    charges.grad.zero_()
    (2.5*erecip).backward()
    drecip2 = charges.grad.clone().detach().numpy()
    assert np.allclose(2.5*ddir, ddir2)
    assert np.allclose(2.5*drecip, drecip2)