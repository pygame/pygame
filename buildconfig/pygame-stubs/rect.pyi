from typing import Dict, List, Sequence, Tuple, TypeVar, Union, overload

from pygame.math import Vector2
from ._common import _Coordinate, _CanBeRect

_K = TypeVar("_K")
_V = TypeVar("_V")

class Rect(object):
    x: int
    y: int
    top: int
    left: int
    bottom: int
    right: int
    topleft: Tuple[int, int]
    bottomleft: Tuple[int, int]
    topright: Tuple[int, int]
    bottomright: Tuple[int, int]
    midtop: Tuple[int, int]
    midleft: Tuple[int, int]
    midbottom: Tuple[int, int]
    midright: Tuple[int, int]
    center: Tuple[int, int]
    centerx: int
    centery: int
    size: Tuple[int, int]
    width: int
    height: int
    w: int
    h: int
    __hash__: None  # type: ignore
    @overload
    def __init__(
        self, left: float, top: float, width: float, height: float
    ) -> None: ...
    @overload
    def __init__(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> None: ...
    @overload
    def __init__(
        self,
        left_top_width_height: Union[
            Rect, Tuple[float, float, float, float], List[float]
        ],
    ) -> None: ...
    @overload
    def __getitem__(self, i: int) -> int: ...
    @overload
    def __getitem__(self, s: slice) -> List[int]: ...
    def copy(self) -> Rect: ...
    @overload
    def move(self, x: float, y: float) -> Rect: ...
    @overload
    def move(self, move_by: _Coordinate) -> Rect: ...
    @overload
    def move_ip(self, x: float, y: float) -> None: ...
    @overload
    def move_ip(self, move_by: _Coordinate) -> None: ...
    @overload
    def inflate(self, x: float, y: float) -> Rect: ...
    @overload
    def inflate(self, inflate_by: _Coordinate) -> Rect: ...
    @overload
    def inflate_ip(self, x: float, y: float) -> None: ...
    @overload
    def inflate_ip(self, inflate_by: _Coordinate) -> None: ...
    @overload
    def update(self, left: float, top: float, width: float, height: float) -> None: ...
    @overload
    def update(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> None: ...
    @overload
    def update(
        self,
        left_top_width_height: Union[
            Rect, Tuple[float, float, float, float], List[float]
        ],
    ) -> None: ...
    @overload
    def clamp(self, rect: Union[_CanBeRect, Rect]) -> Rect: ...
    @overload
    def clamp(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> Rect: ...
    @overload
    def clamp(self, left: float, top: float, width: float, height: float) -> Rect: ...
    @overload
    def clamp_ip(self, rect: Union[_CanBeRect, Rect]) -> None: ...
    @overload
    def clamp_ip(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> None: ...
    @overload
    def clamp_ip(
        self, left: float, top: float, width: float, height: float
    ) -> None: ...
    @overload
    def clip(self, rect: Union[_CanBeRect, Rect]) -> Rect: ...
    @overload
    def clip(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> Rect: ...
    @overload
    def clip(self, left: float, top: float, width: float, height: float) -> Rect: ...
    @overload
    def clipline(
        self, x1: float, x2: float, x3: float, x4: float
    ) -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[()]]: ...
    @overload
    def clipline(
        self, first_coordinate: _Coordinate, second_coordinate: _Coordinate
    ) -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[()]]: ...
    @overload
    def clipline(
        self, values: Union[Tuple[float, float, float, float], List[float]]
    ) -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[()]]: ...
    @overload
    def clipline(
        self, coordinates: Union[Tuple[_Coordinate, _Coordinate], List[_Coordinate]]
    ) -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[()]]: ...
    @overload
    def union(self, rect: Union[_CanBeRect, Rect]) -> Rect: ...
    @overload
    def union(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> Rect: ...
    @overload
    def union(self, left: float, top: float, width: float, height: float) -> Rect: ...
    @overload
    def union_ip(self, rect: Union[_CanBeRect, Rect]) -> None: ...
    @overload
    def union_ip(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> None: ...
    @overload
    def union_ip(
        self, left: float, top: float, width: float, height: float
    ) -> None: ...
    def unionall(self, rect: Sequence[Union[_CanBeRect, Rect]]) -> Rect: ...
    def unionall_ip(self, rect_sequence: Sequence[Union[_CanBeRect, Rect]]) -> None: ...
    @overload
    def fit(self, rect: Union[_CanBeRect, Rect]) -> Rect: ...
    @overload
    def fit(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> Rect: ...
    @overload
    def fit(self, left: float, top: float, width: float, height: float) -> Rect: ...
    def normalize(self) -> None: ...
    @overload
    def __contains__(self, rect: Union[_CanBeRect, Rect, int]) -> bool: ...
    @overload
    def __contains__(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> bool: ...
    @overload
    def __contains__(self, left: float, top: float, width: float, height: float) -> bool: ...
    @overload
    def contains(self, rect: Union[_CanBeRect, Rect]) -> bool: ...
    @overload
    def contains(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> bool: ...
    @overload
    def contains(
        self, left: float, top: float, width: float, height: float
    ) -> bool: ...
    @overload
    def collidepoint(self, x: float, y: float) -> bool: ...
    @overload
    def collidepoint(self, x_y: Union[List[float], Tuple[float, float]]) -> bool: ...
    @overload
    def colliderect(self, rect: Union[_CanBeRect, Rect]) -> bool: ...
    @overload
    def colliderect(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> bool: ...
    @overload
    def colliderect(
        self, left: float, top: float, width: float, height: float
    ) -> bool: ...
    def collidelist(self, rect_list: Sequence[Union[Rect, _CanBeRect]]) -> int: ...
    def collidelistall(
        self, rect_list: Sequence[Union[Rect, _CanBeRect]]
    ) -> List[int]: ...
    # Also undocumented: the dict collision methods take a 'values' argument
    # that defaults to False. If it is False, the keys in rect_dict must be
    # Rect-like; otherwise, the values must be Rects.
    @overload
    def collidedict(
        self, rect_dict: Dict[_CanBeRect, _V], values: bool = ...
    ) -> Tuple[_CanBeRect, _V]: ...
    @overload
    def collidedict(
        self, rect_dict: Dict[_K, "Rect"], values: bool
    ) -> Tuple[_K, "Rect"]: ...
    @overload
    def collidedictall(
        self, rect_dict: Dict[_CanBeRect, _V], values: bool = ...
    ) -> List[Tuple[_CanBeRect, _V]]: ...
    @overload
    def collidedictall(
        self, rect_dict: Dict[_K, "Rect"], values: bool
    ) -> List[Tuple[_K, "Rect"]]: ...
