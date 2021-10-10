from typing import Union, Tuple, List, Sequence
from typing_extensions import Protocol

from pygame.surface import Surface
from pygame.color import Color
from pygame.rect import Rect
from pygame._common import _Coordinate

_ColorValue = Union[
    Color, Tuple[int, int, int], List[int], int, Tuple[int, int, int, int]
]

_CanBeRect = Union[
    Rect,
    Tuple[int, int, int, int], List[int],
    Tuple[_Coordinate, _Coordinate], List[_Coordinate]
]
class _HasRectAttribute(Protocol):
    rect: _CanBeRect
_RectValue = Union[
    _CanBeRect, _HasRectAttribute
]

def pixel(surface: Surface, x: int, y: int, color: _ColorValue) -> None: ...
def hline(surface: Surface, x1: int, x2: int, y: int, color: _ColorValue) -> None: ...
def vline(surface: Surface, x: int, y1: int, y2: int, color: _ColorValue) -> None: ...
def line(
    surface: Surface, x1: int, y1: int, x2: int, y2: int, color: _ColorValue
) -> None: ...
def rectangle(surface: Surface, rect: _RectValue, color: _ColorValue) -> None: ...
def box(surface: Surface, rect: _RectValue, color: _ColorValue) -> None: ...
def circle(surface: Surface, x: int, y: int, r: int, color: _ColorValue) -> None: ...
def aacircle(surface: Surface, x: int, y: int, r: int, color: _ColorValue) -> None: ...
def filled_circle(
    surface: Surface, x: int, y: int, r: int, color: _ColorValue
) -> None: ...
def ellipse(
    surface: Surface, x: int, y: int, rx: int, ry: int, color: _ColorValue
) -> None: ...
def aaellipse(
    surface: Surface, x: int, y: int, rx: int, ry: int, color: _ColorValue
) -> None: ...
def filled_ellipse(
    surface: Surface, x: int, y: int, rx: int, ry: int, color: _ColorValue
) -> None: ...
def arc(
    surface: Surface,
    x: int,
    y: int,
    r: int,
    start_angle: int,
    atp_angle: int,
    color: _ColorValue,
) -> None: ...
def pie(
    surface: Surface,
    x: int,
    y: int,
    r: int,
    start_angle: int,
    atp_angle: int,
    color: _ColorValue,
) -> None: ...
def trigon(
    surface: Surface,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    x3: int,
    y3: int,
    color: _ColorValue,
) -> None: ...
def aatrigon(
    surface: Surface,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    x3: int,
    y3: int,
    color: _ColorValue,
) -> None: ...
def filled_trigon(
    surface: Surface,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    x3: int,
    y3: int,
    color: _ColorValue,
) -> None: ...
def polygon(
    surface: Surface, points: Sequence[_Coordinate], color: _ColorValue
) -> None: ...
def aapolygon(
    surface: Surface, points: Sequence[_Coordinate], color: _ColorValue
) -> None: ...
def filled_polygon(
    surface: Surface, points: Sequence[_Coordinate], color: _ColorValue
) -> None: ...
def textured_polygon(
    surface: Surface, points: Sequence[_Coordinate], texture: Surface, tx: int, ty: int
) -> None: ...
def bezier(
    surface: Surface, points: Sequence[_Coordinate], steps: int, color: _ColorValue
) -> None: ...
