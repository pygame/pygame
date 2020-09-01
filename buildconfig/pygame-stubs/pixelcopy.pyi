import sys
from typing import Optional

import numpy
from . import surface

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_kind = Literal["P", "p", "R", "r", "G", "g", "B", "b", "A", "a", "C", "c"]

def surface_to_array(
    array: numpy.ndarray,
    surface: surface.Surface,
    kind: Optional[_kind] = ...,
    opaque: Optional[int] = ...,
    clear: Optional[int] = ...,
) -> None: ...
def array_to_surface(surface: surface.Surface, array: numpy.ndarray) -> None: ...
def map_to_array(array1: numpy.ndarray, array2: numpy.ndarray, surface: surface.Surface) -> None: ...
def make_surface(array: numpy.ndarray) -> surface.Surface: ...

