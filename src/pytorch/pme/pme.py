from ..neighbors import getNeighborPairs
import torch
import math

class PME:
    def __init__(self, gridx: int, gridy: int, gridz: int, order: int, alpha: float, coulomb: float, exclusions: torch.Tensor):
        self.gridx = gridx
        self.gridy = gridy
        self.gridz = gridz
        self.order = order
        self.alpha = alpha
        self.coulomb = coulomb
        self.exclusions, _ = torch.sort(exclusions.to(torch.int32), descending=True)

        # Initialize the bspline moduli.

        max_size = max(gridx, gridy, gridz)
        data = torch.zeros(order, dtype=torch.float32)
        ddata = torch.zeros(order, dtype=torch.float32)
        bsplines_data = torch.zeros(max_size, dtype=torch.float32)
        data[0] = 1
        for i in range(3, order):
            data[i-1] = 0
            for j in range(1, i-1):
                data[i-j-1] = (j*data[i-j-2]+(i-j)*data[i-j-1])/(i-1)
            data[0] /= i-1

        # Differentiate.

        ddata[0] = -data[0]
        ddata[1:order] = data[0:order-1]-data[1:order]
        for i in range(1, order-1):
            data[order-i-1] = (i*data[order-i-2]+(order-i)*data[order-i-1])/(order-1)
        data[0] /= order-1
        bsplines_data[1:order+1] = data

        # Evaluate the actual bspline moduli for X/Y/Z.

        self.moduli = []
        for ndata in (gridx, gridy, gridz):
            m = torch.zeros(ndata, dtype=torch.float32)
            for i in range(ndata):
                arg = (2*torch.pi*i/ndata)*torch.arange(ndata)
                sc = torch.sum(bsplines_data[:ndata]*torch.cos(arg))
                ss = torch.sum(bsplines_data[:ndata]*torch.sin(arg))
                m[i] = sc*sc + ss*ss
            for i in range(ndata):
                if m[i] < 1e-7:
                    m[i] = (m[(i-1+ndata)%ndata]+m[(i+1)%ndata])*0.5
            self.moduli.append(m)

    def compute_direct(self, positions: torch.Tensor, charges: torch.Tensor, cutoff: float, box_vectors: torch.Tensor, max_num_pairs: int = -1):
        neighbors, deltas, distances, number_found_pairs = getNeighborPairs(positions, cutoff, max_num_pairs, box_vectors)
        self.exclusions = self.exclusions.to(positions.device)
        return torch.ops.pme.pme_direct(positions, charges, neighbors, deltas, distances, self.exclusions, self.alpha, self.coulomb)

    def compute_reciprocal(self, positions: torch.Tensor, charges: torch.Tensor, box_vectors: torch.Tensor):
        for i in range(3):
            self.moduli[i] = self.moduli[i].to(positions.device)
        self_energy = -torch.sum(charges**2)*self.coulomb*self.alpha/math.sqrt(torch.pi)
        return self_energy + torch.ops.pme.pme_reciprocal(positions, charges, box_vectors, self.gridx, self.gridy, self.gridz,
                                            self.order, self.alpha, self.coulomb, self.moduli[0], self.moduli[1], self.moduli[2])
