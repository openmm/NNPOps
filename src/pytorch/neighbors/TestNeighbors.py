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
def test_neighbor_values(device, dtype, num_atoms, cutoff, all_pairs):

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
@pytest.mark.parametrize('num_atoms', [1, 2, 3, 4, 5, 10, 100, 1000])
@pytest.mark.parametrize('grad', ['deltas', 'distances', 'combined'])
def test_neighbor_grads(device, dtype, num_atoms, grad):

    if not pt.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')

    cutoff=1000

    # Generate random positions
    positions = 10 * pt.randn((num_atoms, 3), device=device, dtype=dtype)

    # Compute reference values using pure pytorch
    ref_neighbors = pt.vstack((pt.tril_indices(num_atoms,num_atoms, -1, device=device),))
    ref_positions = positions.clone()
    ref_positions.requires_grad_(True)
    ref_deltas = ref_positions[ref_neighbors[0]] - ref_positions[ref_neighbors[1]]
    ref_distances = pt.linalg.norm(ref_deltas, axis=1)


    # Compute values using NNPOps
    positions.requires_grad_(True)
    print(positions)
    neighbors, deltas, distances = getNeighborPairs(positions, cutoff=cutoff)

    assert pt.all(neighbors > -1)
    assert pt.all(neighbors == ref_neighbors)
    assert pt.allclose(deltas, ref_deltas)
    assert pt.allclose(distances, ref_distances)

    # Compute gradients
    if grad == 'deltas':
        ref_deltas.sum().backward()
        deltas.sum().backward()
    elif grad == 'distances':
        ref_distances.sum().backward()
        distances.sum().backward()
    elif grad == 'combined':
        (ref_deltas.sum() + ref_distances.sum()).backward()
        (deltas.sum() + distances.sum()).backward()
    else:
        raise ValueError('grad')

    if dtype == pt.float32:
        assert pt.allclose(ref_positions.grad, positions.grad, atol=1e-3, rtol=1e-3)
    else:
        assert pt.allclose(ref_positions.grad, positions.grad, atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [pt.float32, pt.float64])
def test_too_many_neighbors(device, dtype):
    if not pt.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    # 4 points result into 6 pairs, but there is a storage just for 4.
    positions = pt.zeros((4, 3,), device=device, dtype=dtype)
    with pytest.raises(RuntimeError):
        # checkErrors = False will throw due to exceeding neighbours
        # syncExceptions = True makes  this exception catchable at the
        # expense of performance (even when no error ocurred)
        getNeighborPairs(positions, cutoff=1, max_num_neighbors=1, check_errors=False, sync_exceptions=True)
        pt.cuda.synchronize()

    # checkErrors = True will never throw due to exceeding neighbours,
    # but  will return  the number  of pairs  found.
    # syncExceptions is ignored in this case
    neighbors, deltas, distances, number_found_pairs = getNeighborPairs(positions, cutoff=1, max_num_neighbors=1, check_errors=True)
    assert number_found_pairs == 6


def test_is_cuda_graph_compatible():
    if not pt.cuda.is_available():
        pytest.skip('No GPU')
    device = 'cuda'
    dtype = pt.float32
    num_atoms = 100
    # Generate random positions
    positions = 10 * pt.randn((num_atoms, 3), device=device, dtype=dtype)
    cutoff = 5
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

    graph = pt.cuda.CUDAGraph()
    s = pt.cuda.Stream()
    s.wait_stream(pt.cuda.current_stream())
    with pt.cuda.stream(s):
        for _ in range(3):
            neighbors, deltas, distances = getNeighborPairs(positions, cutoff=cutoff, max_num_neighbors=num_neighbors+1)
    pt.cuda.synchronize()

    with pt.cuda.graph(graph):
        neighbors, deltas, distances = getNeighborPairs(positions, cutoff=cutoff, max_num_neighbors=num_neighbors+1)

    graph.replay()
    pt.cuda.synchronize()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [pt.float32, pt.float64])
def test_periodic_neighbors(device, dtype):

    if not pt.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')

    # Generate random positions
    num_atoms = 100
    positions = (20 * pt.randn((num_atoms, 3), device=device, dtype=dtype)) - 10
    box_vectors = pt.tensor([[10, 0, 0], [2, 12, 0], [0, 1, 11]], device=device, dtype=dtype)
    cutoff = 5.0

    # Get neighbor pairs
    ref_neighbors = np.vstack(np.tril_indices(num_atoms, -1))
    ref_positions = positions.cpu().numpy()
    ref_vectors = box_vectors.cpu().numpy()
    ref_deltas = ref_positions[ref_neighbors[0]] - ref_positions[ref_neighbors[1]]
    ref_deltas -= np.outer(np.round(ref_deltas[:,2]/ref_vectors[2,2]), ref_vectors[2])
    ref_deltas -= np.outer(np.round(ref_deltas[:,1]/ref_vectors[1,1]), ref_vectors[1])
    ref_deltas -= np.outer(np.round(ref_deltas[:,0]/ref_vectors[0,0]), ref_vectors[0])
    ref_distances = np.linalg.norm(ref_deltas, axis=1)

    # Filter the neighbor pairs
    mask = ref_distances > cutoff
    ref_neighbors[:, mask] = -1
    ref_deltas[mask, :] = np.nan
    ref_distances[mask] = np.nan

    # Find the number of neighbors
    num_neighbors = np.count_nonzero(np.logical_not(np.isnan(ref_distances)))
    max_num_neighbors = max(int(np.ceil(num_neighbors / num_atoms)), 1)

    # Compute results
    neighbors, deltas, distances = getNeighborPairs(positions, cutoff=cutoff, max_num_neighbors=max_num_neighbors, box_vectors=box_vectors)

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

    # Sort the neighbors
    # NOTE: GPU returns the neighbor in a non-deterministic order
    ref_neighbors, ref_deltas, ref_distances = sort_neighbors(ref_neighbors, ref_deltas, ref_distances)
    neighbors, deltas, distances = sort_neighbors(neighbors, deltas, distances)

    # Resize the reference
    ref_neighbors, ref_deltas, ref_distances = resize_neighbors(ref_neighbors, ref_deltas, ref_distances, num_atoms * max_num_neighbors)

    assert np.all(ref_neighbors == neighbors)
    assert np.allclose(ref_deltas, deltas, equal_nan=True)
    assert np.allclose(ref_distances, distances, equal_nan=True)
