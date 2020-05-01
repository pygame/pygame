from typing import Optional
import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
import numpy
from pygame.surface import Surface

_kind = Literal["P", "p", "R", "r", "G", "g", "B", "b", "A", "a", "C", "c"]

def surface_to_array(
    array: numpy.ndarray,
    surface: Surface,
    kind: Optional[_kind] = "P",
    opaque: Optional[int] = 255,
    clear: Optional[int] = 0,
) -> None: ...
def array_to_surface(surface: Surface, array: numpy.ndarray) -> None: ...
def map_to_array(
    array1: numpy.ndarray, array2: numpy.ndarray, surface: Surface
) -> None: ...
def make_surface(array: numpy.ndarray) -> Surface: ...
