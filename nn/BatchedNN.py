import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torchani
from torchani.nn import ANIModel, Ensemble, SpeciesEnergies
from typing import List, Optional, Tuple, Union


class TorchANIBatchedNNs(torch.nn.Module):

    def __init__(self, ensemble: Union[ANIModel, Ensemble], elementSymbols: List[str]):

        super().__init__()

        # Handle the case when the ensemble is just one model
        ensemble = [ensemble] if type(ensemble) == ANIModel else ensemble

        # Extract the weihts and biases of the linear layers
        for ilayer in [0, 2, 4, 6]:
            layers = [[model[symbol][ilayer] for symbol in elementSymbols] for model in ensemble]
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

        vectors = torch.matmul(self.layer0_weights, vectors) + self.layer0_biases # Linear 0
        vectors = F.celu(vectors, alpha=0.1)                                      # CELU   1
        vectors = torch.matmul(self.layer2_weights, vectors) + self.layer2_biases # Linear 2
        vectors = F.celu(vectors, alpha=0.1)                                      # CELU   3
        vectors = torch.matmul(self.layer4_weights, vectors) + self.layer4_biases # Linear 4
        vectors = F.celu(vectors, alpha=0.1)                                      # CELU   5
        vectors = torch.matmul(self.layer6_weights, vectors) + self.layer6_biases # Linear 6

        # Sum: [num_mols, num_atoms, num_models, 1, 1] --> [num_mols, num_models]
        # Mean: [num_mols, num_models] --> [num_mols]
        energies = torch.mean(torch.sum(vectors, (1, 3, 4)), 1)

        return SpeciesEnergies(species, energies)