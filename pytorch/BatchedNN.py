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

import os
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torchani
from torchani.nn import ANIModel, Ensemble, SpeciesConverter, SpeciesEnergies
from typing import List, Optional, Tuple, Union

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))
batchedLinear = torch.ops.NNPOpsBatchedNN.BatchedLinear


class TorchANIBatchedNN(torch.nn.Module):

    def __init__(self, converter: SpeciesConverter, ensemble: Union[ANIModel, Ensemble], atomicNumbers: Tensor):

        super().__init__()

        # Convert atomic numbers to a list of species
        species_list = converter((atomicNumbers, torch.empty(0))).species[0].tolist()

        # Handle the case when the ensemble is just one model
        ensemble = [ensemble] if type(ensemble) == ANIModel else ensemble

        # Convert models to the list of linear layers
        models = [list(model.values()) for model in ensemble]

        # Extract the weihts and biases of the linear layers
        for ilayer in [0, 2, 4, 6]:
            layers = [[model[species][ilayer] for species in species_list] for model in models]
            weights, biases = self.batchLinearLayers(layers)
            self.register_parameter(f'layer{ilayer}_weights', weights)
            self.register_parameter(f'layer{ilayer}_biases', biases)

        # Disable autograd for the parameters
        for parameter in self.parameters():
            parameter.requires_grad = False

    @staticmethod
    def batchLinearLayers(layers: List[List[nn.Linear]]) -> Tuple[nn.Parameter, nn.Parameter]:

        num_models = len(layers)
        num_atoms = len(layers[0])

        # Note: different elements have different size linear layers, so we just find maximum sizes
        #       and pad with zeros.
        max_out = max(layer.out_features for layer in sum(layers, []))
        max_in = max(layer.in_features for layer in sum(layers, []))

        # Copy weights and biases
        weights = torch.zeros((1, num_atoms, num_models, max_out, max_in), dtype=torch.float32)
        biases  = torch.zeros((1, num_atoms, num_models, max_out,      1), dtype=torch.float32)
        for imodel, sublayers in enumerate(layers):
            for iatom, layer in enumerate(sublayers):
                num_out, num_in = layer.weight.shape
                weights[0, iatom, imodel, :num_out, :num_in] = layer.weight
                biases [0, iatom, imodel, :num_out,       0] = layer.bias

        return nn.Parameter(weights), nn.Parameter(biases)

    def forward(self, species_aev: Tuple[Tensor, Tensor]) -> SpeciesEnergies:

        species, aev = species_aev

        # Reshape: [num_mols, num_atoms, num_features] --> [num_mols, num_atoms, 1, num_features, 1]
        vectors = aev.unsqueeze(-2).unsqueeze(-1)

        vectors = batchedLinear(vectors, self.layer0_weights, self.layer0_biases) # Linear 0
        vectors = F.celu(vectors, alpha=0.1)                                      # CELU   1
        vectors = batchedLinear(vectors, self.layer2_weights, self.layer2_biases) # Linear 2
        vectors = F.celu(vectors, alpha=0.1)                                      # CELU   3
        vectors = batchedLinear(vectors, self.layer4_weights, self.layer4_biases) # Linear 4
        vectors = F.celu(vectors, alpha=0.1)                                      # CELU   5
        vectors = batchedLinear(vectors, self.layer6_weights, self.layer6_biases) # Linear 6

        # Sum: [num_mols, num_atoms, num_models, 1, 1] --> [num_mols, num_models]
        # Mean: [num_mols, num_models] --> [num_mols]
        energies = torch.mean(torch.sum(vectors, (1, 3, 4)), 1)

        return SpeciesEnergies(species, energies)