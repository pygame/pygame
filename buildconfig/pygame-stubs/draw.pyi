from typing import Union, Optional, Tuple, List, Sequence
from typing_extensions import Protocol

from pygame.color import Color
from pygame.rect import Rect
from pygame.surface import Surface
from pygame._common import _Coordinate

_ColorValue = Union[
    Color, str, Tuple[int, int, int], List[int], int, Tuple[int, int, int, int]
]
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

def rect(
    surface: Surface,
    color: _ColorValue,
    rect: _RectValue,
    width: int = 0,
    border_radius: int = -1,
    border_top_left_radius: int = -1,
    border_top_right_radius: int = -1,
    border_bottom_left_radius: int = -1,
    border_bottom_right_radius: int = -1,
) -> Rect: ...
def polygon(
    surface: Surface,
    color: _ColorValue,
    points: Sequence[_Coordinate],
    width: int = 0,
) -> Rect: ...
def circle(
    surface: Surface,
    color: _ColorValue,
    center: _Coordinate,
    radius: float,
    width: int = 0,
    draw_top_right: Optional[bool] = None,
    draw_top_left: Optional[bool] = None,
    draw_bottom_left: Optional[bool] = None,
    draw_bottom_right: Optional[bool] = None,
) -> Rect: ...
def ellipse(
    surface: Surface, color: _ColorValue, rect: _RectValue, width: int = 0
) -> Rect: ...
def arc(
    surface: Surface,
    color: _ColorValue,
    rect: _RectValue,
    start_angle: float,
    stop_angle: float,
    width: int = 1,
) -> Rect: ...
def line(
    surface: Surface,
    color: _ColorValue,
    start_pos: _Coordinate,
    end_pos: _Coordinate,
    width: int = 1,
) -> Rect: ...
def lines(
    surface: Surface,
    color: _ColorValue,
    closed: bool,
    points: Sequence[_Coordinate],
    width: int = 1,
) -> Rect: ...
def aaline(
    surface: Surface,
    color: _ColorValue,
    start_pos: _Coordinate,
    end_pos: _Coordinate,
    blend: int = 1,
) -> Rect: ...
def aalines(
    surface: Surface,
    color: _ColorValue,
    closed: bool,
    points: Sequence[_Coordinate],
    blend: int = 1,
) -> Rect: ...
