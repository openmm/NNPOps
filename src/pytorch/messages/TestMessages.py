import pytest
import torch as pt
from NNPOps.messages import passMessages


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [pt.float32, pt.float64])
@pytest.mark.parametrize('num_pairs', [1, 2, 3, 4, 5, 10, 100])
@pytest.mark.parametrize('num_atoms', [1, 2, 3, 4, 5, 10, 100])
@pytest.mark.parametrize('num_states', [32, 64, 1024])
def testPassMessageValues(device, dtype, num_pairs, num_atoms, num_states):

    device = pt.device(device)
    if not pt.cuda.is_available() and device.is_cuda():
        pytest.skip('No GPU')

    # Generate random neighbors
    neighbors = pt.randint(0, num_atoms, (2, num_pairs), dtype=pt.int32, device=device)
    neighbors[:, pt.rand(num_pairs) > 0.5] = -1

    # Generate random messages and states
    messages = pt.randn((num_pairs, num_states), dtype=dtype, device=device)
    states = pt.randn((num_atoms, num_states), dtype=dtype, device=device)

    # Compute reference
    mask = pt.logical_and(neighbors[0] > -1, neighbors[1] > -1)
    masked_neighbors = neighbors[:, mask].to(pt.long)
    masked_messages = messages[mask, :]
    ref_new_states = states.index_add(0, masked_neighbors[0], masked_messages)\
                           .index_add(0, masked_neighbors[1], masked_messages)

    # Compute results
    new_states = passMessages(neighbors, messages, states)

    # Check data type and device
    assert new_states.device == neighbors.device
    assert new_states.dtype == dtype

    # Check values
    if dtype == pt.float32:
        assert pt.allclose(ref_new_states, new_states, atol=1e-6, rtol=1e-4)
    else:
        assert pt.allclose(ref_new_states, new_states, atol=1e-12, rtol=1e-8)

@pytest.mark.parametrize('dtype', [pt.float32, pt.float64])
@pytest.mark.parametrize('num_pairs', [1, 2, 3, 4, 5, 10, 100])
@pytest.mark.parametrize('num_atoms', [1, 2, 3, 4, 5, 10, 100])
@pytest.mark.parametrize('num_states', [32, 64, 1024])
def testPassMessagesGrads(dtype, num_pairs, num_atoms, num_states):

    if not pt.cuda.is_available():
        pytest.skip('No GPU')

    # Generate random neighbors
    neighbors = pt.randint(0, num_atoms, (2, num_pairs), dtype=pt.int32)
    neighbors[:, pt.rand(num_pairs) > 0.5] = -1

    # Generate random messages and states
    messages = pt.randn((num_pairs, num_states), dtype=dtype)
    states = pt.randn((num_atoms, num_states), dtype=dtype)

    # Compute CPU gradients
    neighbors_cpu = neighbors.detach().cpu()
    messages_cpu = messages.detach().cpu()
    states_cpu = states.detach().cpu()
    messages_cpu.requires_grad_()
    states_cpu.requires_grad_()
    passMessages(neighbors_cpu, messages_cpu, states_cpu).norm().backward()

    # Compute CUDA gradients
    neighbors_cuda = neighbors.detach().cuda()
    messages_cuda = messages.detach().cuda()
    states_cuda = states.detach().cuda()
    messages_cuda.requires_grad_()
    states_cuda.requires_grad_()
    passMessages(neighbors_cuda, messages_cuda, states_cuda).norm().backward()

    # Check type and device
    assert messages_cuda.grad.dtype == dtype
    assert states_cuda.grad.dtype == dtype
    assert messages_cuda.grad.device == neighbors_cuda.device
    assert states_cuda.grad.device == neighbors_cuda.device

    # Check gradients
    if dtype == pt.float32:
        assert pt.allclose(messages_cpu.grad, messages_cuda.grad.cpu(), atol=1e-6, rtol=1e-4)
        assert pt.allclose(states_cpu.grad, states_cuda.grad.cpu(), atol=1e-6, rtol=1e-4)
    else:
        assert pt.allclose(messages_cpu.grad, messages_cuda.grad.cpu(), atol=1e-12, rtol=1e-8)
        assert pt.allclose(states_cpu.grad, states_cuda.grad.cpu(), atol=1e-12, rtol=1e-8)