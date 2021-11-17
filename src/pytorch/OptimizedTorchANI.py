#
# Copyright (c) 2020-2021 Acellera
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

import torch
from torch import Tensor
from torchani.models import BuiltinModel
from torchani.utils import SpeciesEnergies
from typing import Optional, Tuple

from NNPOps.BatchedNN import TorchANIBatchedNN
from NNPOps.EnergyShifter import TorchANIEnergyShifter
from NNPOps.SpeciesConverter import TorchANISpeciesConverter
from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions

class OptimizedTorchANI(torch.nn.Module):

    def __init__(self, model: BuiltinModel, atomicNumbers: Tensor) -> None:

        super().__init__()

        # Optimize the components of an ANI model
        self.species_converter = TorchANISpeciesConverter(model.species_converter, atomicNumbers)
        self.aev_computer = TorchANISymmetryFunctions(model.aev_computer)
        self.neural_networks = TorchANIBatchedNN(model.species_converter, model.neural_networks, atomicNumbers)
        self.energy_shifter = TorchANIEnergyShifter(model.species_converter, model.energy_shifter, atomicNumbers)

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:

        species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)
        species_energies = self.energy_shifter(species_energies)

        return species_energies