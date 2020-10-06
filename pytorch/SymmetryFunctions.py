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

from typing import List, Optional, Tuple
import torch
from torch import Tensor
import torchani
from torchani.aev import SpeciesAEV

torch.ops.load_library('libNNPOpsPyTorch.so')

class ANISymmetryFunctions(torch.nn.Module):

    def __init__(self, numSpecies: int,
                              Rcr: float,
                              Rca: float,
                             EtaR: List[float],
                             ShfR: List[float],
                             EtaA: List[float],
                             Zeta: List[float],
                             ShfA: List[float],
                             ShfZ: List[float]):

        super().__init__()

        self.numSpecies = numSpecies
        self.Rcr = Rcr
        self.Rca = Rca
        self.EtaR = EtaR
        self.ShfR = ShfR
        self.EtaA = EtaA
        self.Zeta = Zeta
        self.ShfA = ShfA
        self.ShfZ = ShfZ

    def forward(self, speciesAndPositions: Tuple[Tensor, Tensor],
                      cell: Optional[Tensor] = None,
                      pbc: Optional[Tensor] = None) -> SpeciesAEV:

        species, positions = speciesAndPositions
        if cell and pbc.tolist() != [True, True, True]:
            raise ValueError('Only fully periodic systems are supported, i.e. pbc = [True, True, True]')

        symFunc = torch.ops.NNPOps.ANISymmetryFunctions
        radial, angular = symFunc(self.numSpecies, self.Rcr, self.Rca, self.EtaR, self.ShfR,
                                  self.EtaA, self.Zeta, self.ShfA, self.ShfZ,
                                  species.tolist()[0], positions[0], cell)
        features = torch.cat((radial, angular), dim=1).unsqueeze(0)

        return SpeciesAEV(species, features)

def convertSymmetryFunctions(symmFunc: torchani.AEVComputer) -> ANISymmetryFunctions:

    numSpecies = symmFunc.num_species
    Rcr = symmFunc.Rcr
    Rca = symmFunc.Rca
    EtaR = symmFunc.EtaR[:, 0].tolist()
    ShfR = symmFunc.ShfR[0, :].tolist()
    EtaA = symmFunc.EtaA[:, 0, 0, 0].tolist()
    Zeta = symmFunc.Zeta[0, :, 0, 0].tolist()
    ShfA = symmFunc.ShfA[0, 0, :, 0].tolist()
    ShfZ = symmFunc.ShfZ[0, 0, 0, :].tolist()

    return ANISymmetryFunctions(numSpecies, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ)