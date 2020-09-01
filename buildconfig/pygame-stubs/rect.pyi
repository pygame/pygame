from typing import Iterable, List, Mapping, Sequence, Tuple, TypeVar, Union, overload

from . import math

_K = TypeVar("_K")
_V = TypeVar("_V")

_RectStyle = Union[
    Tuple[float, float, float, float],
    Tuple[Tuple[float, float], Tuple[float, float]],
    List[float],
    List[math.Vector2],
    Tuple[math.Vector2, math.Vector2],
    Iterable[math.Vector2],
]
_Coordinate = Union[Tuple[float, float], Sequence[float], math.Vector2]

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
    def __hash__(self) -> int: ...
    @overload
    def __init__(self, left: float, top: float, width: float, height: float) -> None: ...
    @overload
    def __init__(
        self,
        left_top: Union[Sequence[float], Tuple[float, float], math.Vector2],
        width_height: Union[Sequence[float], Tuple[float, float], math.Vector2],
    ) -> None: ...
    @overload
    def __init__(self, left_top_width_height: Union[Tuple[float, float, float, float], Sequence[float]],) -> None: ...
    @overload
    def __getitem__(self, i: int) -> int: ...
    @overload
    def __getitem__(self, s: slice) -> List[int]: ...
    def copy(self) -> Rect: ...
    def move(self, x: float, y: float) -> Rect: ...
    def move_ip(self, x: float, y: float) -> None: ...
    def inflate(self, x: float, y: float) -> Rect: ...
    def inflate_ip(self, x: float, y: float) -> None: ...
    @overload
    def clamp(self, rect: Union[_RectStyle, Rect]) -> Rect: ...
    @overload
    def clamp(
        self,
        left_top: Union[Sequence[float], Tuple[float, float], math.Vector2],
        width_height: Union[Sequence[float], Tuple[float, float], math.Vector2],
    ) -> Rect: ...
    @overload
    def clamp(self, left: float, top: float, width: float, height: float) -> Rect: ...
    @overload
    def clamp_ip(self, rect: Union[_RectStyle, Rect]) -> None: ...
    @overload
    def clamp_ip(
        self,
        left_top: Union[Sequence[float], Tuple[float, float], math.Vector2],
        width_height: Union[Sequence[float], Tuple[float, float], math.Vector2],
    ) -> None: ...
    @overload
    def clamp_ip(self, left: float, top: float, width: float, height: float) -> None: ...
    @overload
    def clip(self, rect: Union[_RectStyle, Rect]) -> Rect: ...
    @overload
    def clip(
        self,
        left_top: Union[Sequence[float], Tuple[float, float], math.Vector2],
        width_height: Union[Sequence[float], Tuple[float, float], math.Vector2],
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
        self, values: Union[Tuple[float, float, float, float], Sequence[float]]
    ) -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[()]]: ...
    @overload
    def clipline(
        self, coordinates: Union[Tuple[_Coordinate, _Coordinate], Sequence[_Coordinate]]
    ) -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[()]]: ...
    @overload
    def union(self, rect: Union[_RectStyle, Rect]) -> Rect: ...
    @overload
    def union(
        self,
        left_top: Union[Sequence[float], Tuple[float, float], math.Vector2],
        width_height: Union[Sequence[float], Tuple[float, float], math.Vector2],
    ) -> Rect: ...
    @overload
    def union(self, left: float, top: float, width: float, height: float) -> Rect: ...
    @overload
    def union_ip(self, rect: Union[_RectStyle, Rect]) -> None: ...
    @overload
    def union_ip(
        self,
        left_top: Union[Sequence[float], Tuple[float, float], math.Vector2],
        width_height: Union[Sequence[float], Tuple[float, float], math.Vector2],
    ) -> None: ...
    @overload
    def union_ip(self, left: float, top: float, width: float, height: float) -> None: ...
    def unionall(self, rect: Sequence[Union[_RectStyle, Rect]]) -> Rect: ...
    def unionall_ip(self, rect_sequence: Sequence[Union[_RectStyle, Rect]]) -> None: ...
    @overload
    def fit(self, rect: Union[_RectStyle, Rect]) -> Rect: ...
    @overload
    def fit(
        self,
        left_top: Union[Sequence[float], Tuple[float, float], math.Vector2],
        width_height: Union[Sequence[float], Tuple[float, float], math.Vector2],
    ) -> Rect: ...
    @overload
    def fit(self, left: float, top: float, width: float, height: float) -> Rect: ...
    def normalize(self) -> None: ...
    @overload
    def contains(self, rect: Union[_RectStyle, Rect]) -> int: ...
    @overload
    def contains(
        self,
        left_top: Union[Sequence[float], Tuple[float, float], math.Vector2],
        width_height: Union[Sequence[float], Tuple[float, float], math.Vector2],
    ) -> int: ...
    @overload
    def contains(self, left: float, top: float, width: float, height: float) -> int: ...
    @overload
    def collidepoint(self, x: float, y: float) -> int: ...
    @overload
    def collidepoint(self, x_y: Union[Sequence[float], Tuple[float, float]]) -> int: ...
    @overload
    def colliderect(self, rect: Union[_RectStyle, Rect]) -> int: ...
    @overload
    def colliderect(
        self,
        left_top: Union[Sequence[float], Tuple[float, float], math.Vector2],
        width_height: Union[Sequence[float], Tuple[float, float], math.Vector2],
    ) -> int: ...
    @overload
    def colliderect(self, left: float, top: float, width: float, height: float) -> int: ...
    def collidelist(self, rect_list: Sequence[Union[Rect, _RectStyle]]) -> int: ...
    def collidelistall(self, rect_list: Sequence[Union[Rect, _RectStyle]]) -> List[int]: ...
    # Also undocumented: the dict collision methods take a 'values' argument
    # that defaults to False. If it is False, the keys in rect_dict must be
    # Rect-like; otherwise, the values must be Rects.
    @overload
    def collidedict(self, rect_dict: Mapping[_RectStyle, _V], values: bool = ...) -> Tuple[_RectStyle, _V]: ...
    @overload
    def collidedict(self, rect_dict: Mapping[_K, "Rect"], values: bool) -> Tuple[_K, "Rect"]: ...
    @overload
    def collidedictall(self, rect_dict: Mapping[_RectStyle, _V], values: bool = ...) -> List[Tuple[_RectStyle, _V]]: ...
    @overload
    def collidedictall(self, rect_dict: Mapping[_K, "Rect"], values: bool) -> List[Tuple[_K, "Rect"]]: ...

