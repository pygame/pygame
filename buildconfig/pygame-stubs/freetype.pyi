from typing import Tuple, Optional, Union, List, Text, IO, Sequence, Any

from pygame.surface import Surface
from pygame.color import Color
from pygame.rect import Rect

_ColorValue = Union[Color, Tuple[int, int, int], List[int], int]

def get_error() -> str: ...
def get_version() -> Tuple[int, int, int]: ...
def init(cache_size: Optional[int] = 64, resolution: Optional[int] = 72): ...
def quit(): ...
def get_init() -> bool: ...
def was_init() -> bool: ...
def get_cache_size() -> int: ...
def get_default_resolution() -> int: ...
def set_default_resolution(resolution: int) -> None: ...
def SysFont(
    name: Union[str, List[str]],
    size: int,
    bold: Optional[int] = False,
    italic: Optional[int] = False,
): ...
def get_default_font() -> str: ...

STYLE_NORMAL: int
STYLE_UNDERLINE: int
STYLE_OBLIQUE: int
STYLE_STRONG: int
STYLE_WIDE: int
STYLE_DEFAULT: int

class Font:
    name: str
    path: Text
    size: Union[float, Tuple[float, float]]
    height: int
    ascender: int
    descender: int
    style: int
    underline: bool
    strong: bool
    oblique: bool
    wide: bool
    strength: float
    underline_adjustment: float
    fixed_width: bool
    fixed_sizes: int
    scalable: bool
    use_bitmap_strikes: bool
    antialiased: bool
    kerning: bool
    vertical: bool
    rotation: int
    fgcolor: Color
    bgcolor: Color
    origin: bool
    pad: bool
    ucs4: bool
    resolution: int
    def __init__(
        self,
        file: Union[str, IO],
        size: Optional[float] = 0,
        font_index: Optional[int] = 0,
        resolution: Optional[int] = 0,
        ucs4: Optional[int] = False,
    ) -> None: ...
    def get_rect(
        self,
        text: str,
        style: Optional[int] = STYLE_DEFAULT,
        rotation: Optional[int] = 0,
        size: Optional[float] = 0,
    ) -> Rect: ...
    def get_metrics(
        self, text: str, size: Optional[float] = 0
    ) -> List[Tuple[int, int, int, int, float, float]]: ...
    def get_sized_ascender(self, size: float) -> int: ...
    def get_sized_descender(self, size: float) -> int: ...
    def get_sized_height(self, size: float) -> int: ...
    def get_sized_glyph_height(self, size: float) -> int: ...
    def get_sizes(self) -> List[Tuple[int, int, int, float, float]]: ...
    def render(
        self,
        text: str,
        fgcolor: Optional[_ColorValue] = None,
        bgcolor: Optional[_ColorValue] = None,
        style: Optional[int] = STYLE_DEFAULT,
        rotation: Optional[int] = 0,
        size: Optional[float] = 0,
    ) -> Tuple[Surface, Rect]: ...
    def render_to(
        self,
        surf: Surface,
        dest: Union[Tuple[int, int], Sequence[int], Rect],
        text: str,
        fgcolor: Optional[_ColorValue] = None,
        bgcolor: Optional[_ColorValue] = None,
        style: Optional[int] = STYLE_DEFAULT,
        rotation: Optional[int] = 0,
        size: Optional[float] = 0,
    ) -> Rect: ...
    def render_raw(
        self,
        text: str,
        style: Optional[int] = STYLE_DEFAULT,
        rotation: Optional[int] = 0,
        size: Optional[float] = 0,
        invert: Optional[bool] = False,
    ) -> Tuple[bytes, Tuple[int, int]]: ...
    def render_raw_to(
        self,
        array: Any,
        text: str,
        dest: Optional[Union[Tuple[int, int], List[int]]] = None,
        style: Optional[int] = STYLE_DEFAULT,
        rotation: Optional[int] = 0,
        size: Optional[float] = 0,
        invert: Optional[bool] = False,
    ) -> Rect: ...
