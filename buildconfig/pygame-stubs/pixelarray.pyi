from typing import Optional, Sequence, Tuple, Union

from . import color, surface as s

_ColorValue = Union[color.Color, Tuple[int, int, int], Sequence[int], int, Tuple[int, int, int, int]]

class PixelArray:
    surface: s.Surface
    itemsize: int
    ndim: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    def __init__(self, surface: s.Surface) -> None: ...
    def make_surface(self) -> s.Surface: ...
    def replace(
        self,
        color: _ColorValue,
        repcolor: _ColorValue,
        distance: Optional[float] = ...,
        weights: Optional[Sequence[float]] = ...,
    ) -> None: ...
    def extract(
        self, color: _ColorValue, distance: Optional[float] = ..., weights: Optional[Sequence[float]] = ...,
    ) -> PixelArray: ...
    def compare(
        self, array: PixelArray, distance: Optional[float] = ..., weights: Optional[Sequence[float]] = ...,
    ) -> PixelArray: ...
    def transpose(self) -> PixelArray: ...
    def close(self) -> PixelArray: ...

