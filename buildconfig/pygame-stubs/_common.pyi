from os import PathLike
from typing import IO, List, Sequence, Tuple, Union

from typing_extensions import Protocol

from pygame.color import Color
from pygame.math import Vector2
from pygame.rect import Rect

# For functions that take a file name
_AnyPath = Union[str, bytes, PathLike[str], PathLike[bytes]]

# Most pygame functions that take a file argument should be able to handle
# a _FileArg type
_FileArg = Union[_AnyPath, IO[bytes], IO[str]]

_Coordinate = Union[Tuple[float, float], Sequence[float], Vector2]

# This typehint is used when a function would return an RGBA tuble
_RgbaOutput = Tuple[int, int, int, int]
_ColorValue = Union[Color, int, str, Tuple[int, int, int], List[int], _RgbaOutput]

_CanBeRect = Union[
    Rect,
    Tuple[int, int, int, int],
    List[int],
    Tuple[_Coordinate, _Coordinate],
    List[_Coordinate],
]

class _HasRectAttribute(Protocol):
    rect: _CanBeRect

_RectValue = Union[_CanBeRect, _HasRectAttribute]
