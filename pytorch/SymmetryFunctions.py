#
# Copyright (c) 2020 Acellera
# Authors: Raimondas Galvelis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os.path
from typing import List, Optional, Tuple
import torch
from torch import Tensor
import torchani
from torchani.aev import SpeciesAEV

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))
torch.classes.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))

Holder = torch.classes.NNPOpsANISymmetryFunctions.Holder
operation = torch.ops.NNPOpsANISymmetryFunctions.operation

class TorchANISymmetryFunctions(torch.nn.Module):
    """Optimized TorchANI symmetry functions

    Optimized drop-in replacement for torchani.AEVComputer (https://aiqm.github.io/torchani/api.html?highlight=speciesaev#torchani.AEVComputer)

    Example::

        >>> import mdtraj
        >>> import torch
        >>> import torchani

        >>> from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions

        >>> device = torch.device('cuda')

        # Load a molecule
        >>> molecule = mdtraj.load('molecule.mol2')
        >>> species = torch.tensor([[atom.element.atomic_number for atom in molecule.top.atoms]], device=device)
        >>> positions = torch.tensor(molecule.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)

        # Construct ANI-2x and replace its native featurizer with NNPOps implementation
        >>> nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
        >>> nnp.aev_computer = TorchANISymmetryFunctions(nnp.aev_computer)

        # Compute energy
        >>> energy = nnp((species, positions)).energies
        >>> energy.backward()
        >>> forces = -positions.grad.clone()

        >>> print(energy, forces)
    """
    holder: Optional[Holder]

    def __init__(self, symmFunc: torchani.AEVComputer):
        """
        Arguments:
            symmFunc: the instance of torchani.AEVComputer (https://aiqm.github.io/torchani/api.html#torchani.AEVComputer)
        """
        super().__init__()

        self.numSpecies = symmFunc.num_species
        self.Rcr = symmFunc.Rcr
        self.Rca = symmFunc.Rca
        self.EtaR = symmFunc.EtaR[:, 0].tolist()
        self.ShfR = symmFunc.ShfR[0, :].tolist()
        self.EtaA = symmFunc.EtaA[:, 0, 0, 0].tolist()
        self.Zeta = symmFunc.Zeta[0, :, 0, 0].tolist()
        self.ShfA = symmFunc.ShfA[0, 0, :, 0].tolist()
        self.ShfZ = symmFunc.ShfZ[0, 0, 0, :].tolist()

        self.holder = None

        self.triu_index = torch.tensor([0]) # A dummy variable to make TorchScript happy ;)

    def forward(self, speciesAndPositions: Tuple[Tensor, Tensor],
                      cell: Optional[Tensor] = None,
                      pbc: Optional[Tensor] = None) -> SpeciesAEV:
        """Compute the atomic environment vectors

        The signature of the method is identical to torchani.AEVComputer.forward (https://aiqm.github.io/torchani/api.html?highlight=speciesaev#torchani.AEVComputer.forward)

        Arguments:
            speciesAndPositions: atomic species and positions
            cell: unitcell vectors
            pbc: periodic boundary conditions

        Returns:
            SpeciesAEV: atomic species and environment vectors

        """
        species, positions = speciesAndPositions
        if species.shape[0] != 1:
            raise ValueError('Batched molecule computation is not supported')
        if species.shape + (3,) != positions.shape:
            raise ValueError('Inconsistent shapes of "species" and "positions"')
        if cell is not None:
            if cell.shape != (3, 3):
                raise ValueError('"cell" shape has to be [3, 3]')
            if pbc is None:
                raise ValueError('"pbc" has to be defined')
            else:
                pbc_: List[bool] = pbc.tolist() # Explicit type casting for TorchScript
                if pbc_ != [True, True, True]:
                    raise ValueError('Only fully periodic systems are supported, i.e. pbc = [True, True, True]')

        if self.holder is None:
            species_: List[int] = species[0].tolist() # Explicit type casting for TorchScript
            self.holder = Holder(self.numSpecies, self.Rcr, self.Rca,
                                 self.EtaR, self.ShfR,
                                 self.EtaA, self.Zeta, self.ShfA, self.ShfZ,
                                 species_, positions)

        radial, angular = operation(self.holder, positions[0], cell)
        features = torch.cat((radial, angular), dim=1).unsqueeze(0)

        return SpeciesAEV(species, features)