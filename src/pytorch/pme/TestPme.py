import torch
import pytest
import numpy as np
from NNPOps.pme import PME

class PmeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pme = PME(14, 15, 16, 5, 4.985823141035867, 138.935, torch.zeros(9, 0, dtype=torch.int32))

    def forward(self, positions, charges, box_vectors):
        edir = self.pme.compute_direct(positions, charges, 0.5, box_vectors)
        erecip = self.pme.compute_reciprocal(positions, charges, box_vectors)
        return edir-erecip

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_rectangular(device):
    """Test PME on a rectangular box."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    pme = PME(14, 15, 16, 5, 4.985823141035867, 138.935, torch.zeros(9, 0, dtype=torch.int32))
    pos = [[0.7713206433, 0.02075194936, 0.6336482349],
           [0.7488038825, 0.4985070123, 0.2247966455],
           [0.1980628648, 0.7605307122, 0.1691108366],
           [0.08833981417, 0.6853598184, 0.9533933462],
           [0.003948266328, 0.5121922634, 0.8126209617],
           [0.6125260668, 0.7217553174, 0.2918760682],
           [0.9177741225, 0.7145757834, 0.542544368],
           [0.1421700476, 0.3733407601, 0.6741336151],
           [0.4418331744, 0.4340139933, 0.6177669785]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0, 1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)

    # Compare forces and energies to values computed with OpenMM.

    edirect = pme.compute_direct(positions, charges, 0.5, box_vectors)
    assert np.allclose(0.5811535194516182, edirect.detach().cpu().numpy())
    erecip = pme.compute_reciprocal(positions, charges, box_vectors)
    assert np.allclose(-90.92361028496651, erecip.detach().cpu().numpy())
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
    assert np.allclose(expected_ddirect, positions.grad.cpu().numpy(), rtol=1e-4)
    positions.grad.zero_()
    erecip.backward()
    assert np.allclose(expected_drecip, positions.grad.cpu().numpy(), rtol=1e-4)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_triclinic(device):
    """Test PME on a triclinic box."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    pme = PME(14, 16, 15, 5, 5.0, 138.935, torch.zeros(9, 0, dtype=torch.int32))
    pos = [[1.31396193, -0.9377441519, 0.9009447048],
           [1.246411648, 0.4955210369, -0.3256100634],
           [-0.4058114057, 1.281592137, -0.4926674903],
           [-0.7349805575, 1.056079455, 1.860180039],
           [-0.988155201, 0.5365767902, 1.437862885],
           [0.8375782005, 1.165265952, -0.1243717955],
           [1.753322368, 1.14372735, 0.627633104],
           [-0.5734898572, 0.1200222802, 1.022400845],
           [0.3254995233, 0.30204198, 0.8533009354]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [-0.1, 1.2, 0], [0.2, -0.15, 1.1]], dtype=torch.float32, device=device)

    # Compare forces and energies to values computed with OpenMM.

    edirect = pme.compute_direct(positions, charges, 0.5, box_vectors)
    assert np.allclose(-178.86083489656448, edirect.detach().cpu().numpy())
    erecip = pme.compute_reciprocal(positions, charges, box_vectors)
    assert np.allclose(-200.9420623172533, erecip.detach().cpu().numpy())
    expected_ddirect = [[-1000.97644, -326.2085571, 373.3143005],
                        [401.765686, 153.7181702, -278.0073242],
                        [2140.490723, -633.4395752, -1059.523071],
                        [-1.647740602, 10.02025795, 0.2182842493],
                        [-0, -0, -0],
                        [0.05209997296, -2.530653, 3.196420431],
                        [-2139.176758, 633.9973145, 1060.562622],
                        [13.49786377, 11.52490139, -10.12783146],
                        [585.994812, 152.9181519, -89.63345337]]
    expected_drecip = [[-162.9051514, 32.17734528, -77.43495178],
                       [11.11517906, 52.98329163, -83.18161011],
                       [34.50453186, 8.428194046, -4.691772938],
                       [-12.71308613, 20.7514267, -13.68377304],
                       [-0, -0, -0],
                       [8.277475357, -3.927520275, 13.88403988],
                       [-34.93006897, -7.739934444, 8.986465454],
                       [45.33776474, -36.9358139, 40.34444809],
                       [111.2698975, -65.63329315, 115.8478012]]
    edirect.backward()
    assert np.allclose(expected_ddirect, positions.grad.cpu().numpy(), rtol=1e-4)
    positions.grad.zero_()
    erecip.backward()
    assert np.allclose(expected_drecip, positions.grad.cpu().numpy(), rtol=1e-4)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_exclusions(device):
    """Test PME with exclusions."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    pos = [[1.31396193, -0.9377441519, 0.9009447048],
           [1.246411648, 0.4955210369, -0.3256100634],
           [-0.4058114057, 1.281592137, -0.4926674903],
           [-0.7349805575, 1.056079455, 1.860180039],
           [-0.988155201, 0.5365767902, 1.437862885],
           [0.8375782005, 1.165265952, -0.1243717955],
           [1.753322368, 1.14372735, 0.627633104],
           [-0.5734898572, 0.1200222802, 1.022400845],
           [0.3254995233, 0.30204198, 0.8533009354]]
    excl = [[3, -1],
            [-1, -1],
            [-1, 3],
            [0, 2],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, 8],
            [7, -1]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    exclusions = torch.tensor(excl, dtype=torch.int32, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [-0.1, 1.2, 0], [0.2, -0.15, 1.1]], dtype=torch.float32, device=device)
    pme = PME(14, 16, 15, 5, 5.0, 138.935, exclusions)

    # Compare forces and energies to values computed with OpenMM.

    edirect = pme.compute_direct(positions, charges, 0.5, box_vectors)
    assert np.allclose(-204.22671127319336, edirect.detach().cpu().numpy())
    erecip = pme.compute_reciprocal(positions, charges, box_vectors)
    assert np.allclose(-200.9420623172533, erecip.detach().cpu().numpy())
    expected_ddirect = [[-998.2406773, -314.4639407, 379.7956738],
                        [401.7656421, 153.7181283, -278.0072042],
                        [2136.789297, -634.4331203, -1062.13192],
                        [-0.6838558404, -0.7345126528, -3.655667043],
                        [-0, -0, -0],
                        [0.05210044985, -2.530651058, 3.196419874],
                        [-2139.175743, 634.0007806, 1060.564263],
                        [21.9532636, -40.74009123, 38.42738517],
                        [577.5399728, 205.183407, -138.1889512]]
    expected_drecip = [[-162.9051514, 32.17734528, -77.43495178],
                       [11.11517906, 52.98329163, -83.18161011],
                       [34.50453186, 8.428194046, -4.691772938],
                       [-12.71308613, 20.7514267, -13.68377304],
                       [-0, -0, -0],
                       [8.277475357, -3.927520275, 13.88403988],
                       [-34.93006897, -7.739934444, 8.986465454],
                       [45.33776474, -36.9358139, 40.34444809],
                       [111.2698975, -65.63329315, 115.8478012]]
    edirect.backward()
    assert np.allclose(expected_ddirect, positions.grad.cpu().numpy(), rtol=1e-4)
    positions.grad.zero_()
    erecip.backward()
    assert np.allclose(expected_drecip, positions.grad.cpu().numpy(), rtol=1e-4)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_charge_deriv(device):
    """Test derivatives with respect to charge."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    pos = [[0.7713206433, 0.02075194936, 0.6336482349],
           [0.7488038825, 0.4985070123, 0.2247966455],
           [0.1980628648, 0.7605307122, 0.1691108366],
           [0.08833981417, 0.6853598184, 0.9533933462],
           [0.003948266328, 0.5121922634, 0.8126209617],
           [0.6125260668, 0.7217553174, 0.2918760682],
           [0.9177741225, 0.7145757834, 0.542544368],
           [0.1421700476, 0.3733407601, 0.6741336151],
           [0.4418331744, 0.4340139933, 0.6177669785]]
    excl = [[6, -1],
            [-1, -1],
            [-1, -1],
            [6, -1],
            [-1, -1],
            [-1, -1],
            [0, 3],
            [-1, -1],
            [-1, -1]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    exclusions = torch.tensor(excl, dtype=torch.int32, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, requires_grad=True, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    pme = PME(14, 15, 16, 5, 4.985823141035867, 138.935, exclusions)

    # Compute derivatives of the energies with respect to charges.

    edir = pme.compute_direct(positions, charges, 0.5, box_vectors)
    erecip = pme.compute_reciprocal(positions, charges, box_vectors)
    edir.backward(retain_graph=True)
    ddir = charges.grad.clone().detach().cpu().numpy()
    charges.grad.zero_()
    erecip.backward(retain_graph=True)
    drecip = charges.grad.clone().detach().cpu().numpy()

    # Compute finite difference approximations from two displaced inputs.

    delta = 0.001
    for i in range(len(charges)):
        c1 = charges.clone()
        c1[i] += delta
        edir1 = pme.compute_direct(positions, c1, 0.5, box_vectors).detach().cpu().numpy()
        erecip1 = pme.compute_reciprocal(positions, c1, box_vectors).detach().cpu().numpy()
        c2 = charges.clone()
        c2[i] -= delta
        edir2 = pme.compute_direct(positions, c2, 0.5, box_vectors).detach().cpu().numpy()
        erecip2 = pme.compute_reciprocal(positions, c2, box_vectors).detach().cpu().numpy()
        assert np.allclose(ddir[i], (edir1-edir2)/(2*delta), rtol=1e-3, atol=1e-3)
        assert np.allclose(drecip[i], (erecip1-erecip2)/(2*delta), rtol=1e-3, atol=1e-3)

    # Make sure the chain rule is applied properly.

    charges.grad.zero_()
    (2.5*edir).backward()
    ddir2 = charges.grad.clone().detach().cpu().numpy()
    charges.grad.zero_()
    (2.5*erecip).backward()
    drecip2 = charges.grad.clone().detach().cpu().numpy()
    assert np.allclose(2.5*ddir, ddir2)
    assert np.allclose(2.5*drecip, drecip2)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_jit(device):
    """Test that the model can be JIT compiled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    m1 = PmeModule()
    m2 = torch.jit.script(m1)
    torch.manual_seed(10)
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device)-1
    positions.requires_grad_()
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    e1 = m1(positions, charges, box_vectors)
    e2 = m2(positions, charges, box_vectors)
    assert np.allclose(e1.detach().cpu().numpy(), e2.detach().cpu().numpy())
    e1.backward()
    d1 = positions.grad.detach().cpu().numpy()
    positions.grad.zero_()
    e2.backward()
    d2 = positions.grad.detach().cpu().numpy()
    assert np.allclose(d1, d2)

def test_cuda_graph():
    """Test that PME works with CUDA graphs."""
    if not torch.cuda.is_available():
        pytest.skip('No GPU')
    device = 'cuda'
    pme =  PmeModule()
    torch.manual_seed(10)
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device)-1
    positions.requires_grad_()
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)

    # Warmup before capturing graph.

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            e = pme(positions, charges, box_vectors)
            e.backward()
    torch.cuda.current_stream().wait_stream(s)

    # Capture the graph.

    g = torch.cuda.CUDAGraph()
    positions.grad.zero_()
    with torch.cuda.graph(g):
        e = pme(positions, charges, box_vectors)
        e.backward()

    # Replay the graph.

    g.replay()
    torch.cuda.synchronize()