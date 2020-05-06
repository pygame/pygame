from typing import Union, Optional, Tuple, List, Sequence
from pygame.color import Color
from pygame.rect import Rect
from pygame.surface import Surface
from pygame.math import Vector2

_Coordinate = Union[Tuple[float, float], List[float], Vector2]
_ColorValue = Union[
    Color, Tuple[int, int, int], List[int], int, Tuple[int, int, int, int]
]
_RectValue = Union[
    Rect,
    Union[Tuple[int, int, int, int], List[int]],
    Union[Tuple[_Coordinate, _Coordinate], List[_Coordinate]],
]

def rect(
    surface: Surface,
    color: _ColorValue,
    rect: _RectValue,
    width: Optional[int] = 0,
    border_radius: Optional[int] = -1,
    border_top_left_radius: Optional[int] = -1,
    border_top_right_radius: Optional[int] = -1,
    border_bottom_left_radius: Optional[int] = -1,
    border_bottom_right_radius: Optional[int] = -1,
) -> Rect: ...
def polygon(
    surface: Surface,
    color: _ColorValue,
    points: Sequence[_Coordinate],
    width: Optional[int] = 0,
) -> Rect: ...
def circle(
    surface: Surface,
    color: _ColorValue,
    center: _Coordinate,
    radius: float,
    width: Optional[int] = 0,
    draw_top_right: Optional[bool] = None,
    draw_top_left: Optional[bool] = None,
    draw_bottom_left: Optional[bool] = None,
    draw_bottom_right: Optional[bool] = None,
) -> Rect: ...
def ellipse(
    surface: Surface, color: _ColorValue, rect: _RectValue, width: Optional[int] = 0
) -> Rect: ...
def arc(
    surface: Surface,
    color: _ColorValue,
    rect: _RectValue,
    start_angle: float,
    stop_angle: float,
    width: Optional[int] = 1,
) -> Rect: ...
def line(
    surface: Surface,
    color: _ColorValue,
    start_pos: _Coordinate,
    end_pos: _Coordinate,
    width: Optional[int] = 1,
) -> Rect: ...
def lines(
    surface: Surface,
    color: _ColorValue,
    closed: bool,
    points: Sequence[_Coordinate],
    width: Optional[int] = 1,
) -> Rect: ...
def aaline(
    surface: Surface,
    color: _ColorValue,
    start_pos: _Coordinate,
    end_pos: _Coordinate,
    blend: Optional[int] = 1,
) -> Rect: ...
def aalines(
    surface: Surface,
    color: _ColorValue,
    closed: bool,
    points: Sequence[_Coordinate],
    blend: Optional[int] = 1,
) -> Rect: ...
