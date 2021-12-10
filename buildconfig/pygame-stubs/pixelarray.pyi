from typing import Sequence, Tuple

from pygame.surface import Surface

from ._common import _ColorValue

class PixelArray:
    surface: Surface
    itemsize: int
    ndim: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    def __init__(self, surface: Surface) -> None: ...
    def make_surface(self) -> Surface: ...
    def replace(
        self,
        color: _ColorValue,
        repcolor: _ColorValue,
        distance: float = 0,
        weights: Sequence[float] = (0.299, 0.587, 0.114),
    ) -> None: ...
    def extract(
        self,
        color: _ColorValue,
        distance: float = 0,
        weights: Sequence[float] = (0.299, 0.587, 0.114),
    ) -> PixelArray: ...
    def compare(
        self,
        array: PixelArray,
        distance: float = 0,
        weights: Sequence[float] = (0.299, 0.587, 0.114),
    ) -> PixelArray: ...
    def transpose(self) -> PixelArray: ...
    def close(self) -> PixelArray: ...
