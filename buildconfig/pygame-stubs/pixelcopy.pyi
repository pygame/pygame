import sys
from typing import Optional

from numpy import ndarray
from pygame.surface import Surface

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_kind = Literal["P", "p", "R", "r", "G", "g", "B", "b", "A", "a", "C", "c"]

def surface_to_array(
    array: ndarray, surface: Surface, kind: Optional[_kind] = ..., opaque: Optional[int] = ..., clear: Optional[int] = ...,
) -> None: ...
def array_to_surface(surface: Surface, array: ndarray) -> None: ...
def map_to_array(array1: ndarray, array2: ndarray, surface: Surface) -> None: ...
def make_surface(array: ndarray) -> Surface: ...

