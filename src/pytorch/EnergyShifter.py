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
from torchani.nn import SpeciesConverter
from torchani.utils import EnergyShifter, SpeciesEnergies
from typing import Optional, Tuple


class TorchANIEnergyShifter(torch.nn.Module):

    def __init__(self, converter: SpeciesConverter, shifter: EnergyShifter, atomicNumbers: Tensor) -> None:

        super().__init__()

        # Convert atomic numbers to a list of species
        species_list = converter((atomicNumbers, torch.empty(0))).species[0]
        species_list = torch.reshape(species_list, (1, -1))

        # Compute atomic self energies
        self_energies = torch.nn.Parameter(shifter.sae(species_list))
        self.register_parameter('self_energies', self_energies)

    def forward(self, species_energies: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:

        species, energies = species_energies

        return SpeciesEnergies(species, energies + self.self_energies)