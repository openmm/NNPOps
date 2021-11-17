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
from torchani.nn import SpeciesCoordinates, SpeciesConverter
from typing import Optional, Tuple

class TorchANISpeciesConverter(torch.nn.Module):

    def __init__(self, converter: SpeciesConverter, atomicNumbers: Tensor) -> None:

        super().__init__()

        # Convert atomic numbers to a list of species
        species = converter((atomicNumbers, torch.empty(0))).species
        self.register_buffer('species', species)

        self.conv_tensor = converter.conv_tensor # Just to make TorchScript happy :)

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesCoordinates:

        _, coordinates = species_coordinates

        return SpeciesCoordinates(self.species, coordinates)
