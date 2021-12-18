from os import PathLike
from typing import IO, List, Sequence, Tuple, Union

from typing_extensions import Protocol

from pygame.color import Color
from pygame.math import Vector2
from pygame.rect import Rect

# For functions that take a file name
# Same definition has to be in __init__.pyi
AnyPath = Union[str, bytes, PathLike[str], PathLike[bytes]]

# Most pygame functions that take a file argument should be able to handle
# a FileArg type
FileArg = Union[AnyPath, IO[bytes], IO[str]]

Coordinate = Union[Tuple[float, float], Sequence[float], Vector2]

# This typehint is used when a function would return an RGBA tuble
RgbaOutput = Tuple[int, int, int, int]
ColorValue = Union[Color, int, str, Tuple[int, int, int], List[int], RgbaOutput]

CanBeRect = Union[
    Rect,
    Tuple[int, int, int, int],
    List[int],
    Tuple[Coordinate, Coordinate],
    List[Coordinate],
]

class HasRectAttribute(Protocol):
    rect: CanBeRect

RectValue = Union[CanBeRect, HasRectAttribute]
