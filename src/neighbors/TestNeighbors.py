import numpy as np
import pytest
import torch as pt
from NNPOps import getNeighborPairs


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [pt.float32, pt.float64])
@pytest.mark.parametrize('num_atoms', [1, 2, 3, 4, 5, 10, 100, 1000, 10000])
@pytest.mark.parametrize('cutoff', [1, 10, 100])
def test_neighbors(device, dtype, num_atoms, cutoff):

    if not pt.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')

    # Generate random positions
    positions = 10 * pt.randn((num_atoms, 3), device=device, dtype=dtype)

    # Get neighbor pairs
    ref_neighbors = np.vstack(np.tril_indices(num_atoms, -1))
    ref_positions = positions.cpu().numpy()
    ref_distances = np.linalg.norm(ref_positions[ref_neighbors[0]] - ref_positions[ref_neighbors[1]], axis=1)

    # Filter the neighbor pairs
    mask = ref_distances > cutoff
    ref_neighbors[:, mask] = -1
    ref_distances[mask] = np.nan

    # Compute results
    neighbors, distances = getNeighborPairs(positions, cutoff=cutoff, max_num_neighbors=-1)

    # Check device
    assert neighbors.device == positions.device
    assert distances.device == positions.device

    # Check types
    assert neighbors.dtype == pt.int32
    assert distances.dtype == dtype

    # Covert the results
    neighbors = neighbors.cpu().numpy()
    distances = distances.cpu().numpy()

    assert np.all(ref_neighbors == neighbors)
    assert np.allclose(ref_distances, distances, equal_nan=True)