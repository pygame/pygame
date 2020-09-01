from typing import Optional, Sequence, Tuple, Union

from . import color, math, surface
from .rect import Rect

_Coordinate = Union[Tuple[float, float], Sequence[float], math.Vector2]
_ColorValue = Union[color.Color, str, Tuple[int, int, int], Sequence[int], int, Tuple[int, int, int, int]]
_RectValue = Union[
    Rect, Union[Tuple[int, int, int, int], Sequence[int]], Union[Tuple[_Coordinate, _Coordinate], Sequence[_Coordinate]],
]

def rect(
    surface: surface.Surface,
    color: _ColorValue,
    rect: _RectValue,
    width: Optional[int] = ...,
    border_radius: Optional[int] = ...,
    border_top_left_radius: Optional[int] = ...,
    border_top_right_radius: Optional[int] = ...,
    border_bottom_left_radius: Optional[int] = ...,
    border_bottom_right_radius: Optional[int] = ...,
) -> Rect: ...
def polygon(surface: surface.Surface, color: _ColorValue, points: Sequence[_Coordinate], width: Optional[int] = ...) -> Rect: ...
def circle(
    surface: surface.Surface,
    color: _ColorValue,
    center: _Coordinate,
    radius: float,
    width: Optional[int] = ...,
    draw_top_right: Optional[bool] = ...,
    draw_top_left: Optional[bool] = ...,
    draw_bottom_left: Optional[bool] = ...,
    draw_bottom_right: Optional[bool] = ...,
) -> Rect: ...
def ellipse(surface: surface.Surface, color: _ColorValue, rect: _RectValue, width: Optional[int] = ...) -> Rect: ...
def arc(
    surface: surface.Surface,
    color: _ColorValue,
    rect: _RectValue,
    start_angle: float,
    stop_angle: float,
    width: Optional[int] = ...,
) -> Rect: ...
def line(
    surface: surface.Surface, color: _ColorValue, start_pos: _Coordinate, end_pos: _Coordinate, width: Optional[int] = ...
) -> Rect: ...
def lines(
    surface: surface.Surface, color: _ColorValue, closed: bool, points: Sequence[_Coordinate], width: Optional[int] = ...
) -> Rect: ...
def aaline(
    surface: surface.Surface, color: _ColorValue, start_pos: _Coordinate, end_pos: _Coordinate, blend: Optional[int] = ...
) -> Rect: ...
def aalines(
    surface: surface.Surface, color: _ColorValue, closed: bool, points: Sequence[_Coordinate], blend: Optional[int] = ...
) -> Rect: ...
