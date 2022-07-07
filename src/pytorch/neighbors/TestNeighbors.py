import numpy as np
import pytest
import torch as pt
from NNPOps.neighbors import getNeighborPairs


def sort_neighbors(neighbors, deltas, distances):
    i_sorted = np.lexsort(neighbors)[::-1]
    return neighbors[:, i_sorted], deltas[i_sorted], distances[i_sorted]

def resize_neighbors(neighbors, deltas, distances, num_neighbors):

    new_neighbors = np.full((2, num_neighbors), -1, dtype=neighbors.dtype)
    new_deltas = np.full((num_neighbors, 3), np.nan, dtype=deltas.dtype)
    new_distances = np.full((num_neighbors), np.nan, dtype=distances.dtype)

    if num_neighbors < neighbors.shape[1]:
        assert np.all(neighbors[:, num_neighbors:] == -1)
        assert np.all(np.isnan(deltas[num_neighbors:]))
        assert np.all(np.isnan(distances[num_neighbors:]))
        new_neighbors = neighbors[:, :num_neighbors]
        new_deltas = deltas[:num_neighbors]
        new_distances = distances[:num_neighbors]
    else:
        num_neighbors = neighbors.shape[1]
        new_neighbors[:, :num_neighbors] = neighbors
        new_deltas[:num_neighbors] = deltas
        new_distances[:num_neighbors] = distances

    return new_neighbors, new_deltas, new_distances

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [pt.float32, pt.float64])
@pytest.mark.parametrize('num_atoms', [1, 2, 3, 4, 5, 10, 100, 1000])
@pytest.mark.parametrize('cutoff', [1, 10, 100])
@pytest.mark.parametrize('all_pairs', [True, False])
def test_neighbors(device, dtype, num_atoms, cutoff, all_pairs):

    if not pt.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')

    # Generate random positions
    positions = 10 * pt.randn((num_atoms, 3), device=device, dtype=dtype)

    # Get neighbor pairs
    ref_neighbors = np.vstack(np.tril_indices(num_atoms, -1))
    ref_positions = positions.cpu().numpy()
    ref_deltas = ref_positions[ref_neighbors[0]] - ref_positions[ref_neighbors[1]]
    ref_distances = np.linalg.norm(ref_deltas, axis=1)

    # Filter the neighbor pairs
    mask = ref_distances > cutoff
    ref_neighbors[:, mask] = -1
    ref_deltas[mask, :] = np.nan
    ref_distances[mask] = np.nan

    # Find the number of neighbors
    num_neighbors = np.count_nonzero(np.logical_not(np.isnan(ref_distances)))
    max_num_neighbors = -1 if all_pairs else max(int(np.ceil(num_neighbors / num_atoms)), 1)

    # Compute results
    neighbors, deltas, distances = getNeighborPairs(positions, cutoff=cutoff, max_num_neighbors=max_num_neighbors)

    # Check device
    assert neighbors.device == positions.device
    assert deltas.device == positions.device
    assert distances.device == positions.device

    # Check types
    assert neighbors.dtype == pt.int32
    assert deltas.dtype == dtype
    assert distances.dtype == dtype

    # Covert the results
    neighbors = neighbors.cpu().numpy()
    deltas = deltas.cpu().numpy()
    distances = distances.cpu().numpy()

    if not all_pairs:
        # Sort the neighbors
        # NOTE: GPU returns the neighbor in a non-deterministic order
        ref_neighbors, ref_deltas, ref_distances = sort_neighbors(ref_neighbors, ref_deltas, ref_distances)
        neighbors, deltas, distances = sort_neighbors(neighbors, deltas, distances)

        # Resize the reference
        ref_neighbors, ref_deltas, ref_distances = resize_neighbors(ref_neighbors, ref_deltas, ref_distances, num_atoms * max_num_neighbors)

    assert np.all(ref_neighbors == neighbors)
    assert np.allclose(ref_deltas, deltas, equal_nan=True)
    assert np.allclose(ref_distances, distances, equal_nan=True)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [pt.float32, pt.float64])
def test_too_many_neighbors(device, dtype):

    if not pt.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')

    # 4 points result into 6 pairs, but there is a storage just for 4.
    with pytest.raises(RuntimeError):
        positions = pt.zeros((4, 3,), device=device, dtype=dtype)
        getNeighborPairs(positions, cutoff=1, max_num_neighbors=1)
        pt.cuda.synchronize()