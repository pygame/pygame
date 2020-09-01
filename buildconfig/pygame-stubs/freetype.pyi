from typing import Any, IO, List, Optional, Sequence, Text, Tuple, Union

from . import color, rect, surface

_ColorValue = Union[color.Color, Tuple[int, int, int], Sequence[int], int]

def get_error() -> str: ...
def get_version() -> Tuple[int, int, int]: ...
def init(cache_size: Optional[int] = ..., resolution: Optional[int] = ...) -> None: ...
def quit() -> None: ...
def get_init() -> bool: ...
def was_init() -> bool: ...
def get_cache_size() -> int: ...
def get_default_resolution() -> int: ...
def set_default_resolution(resolution: int) -> None: ...
def SysFont(name: Union[str, Sequence[str]], size: int, bold: Optional[int] = ..., italic: Optional[int] = ...,) -> Font: ...
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
    fgcolor: color.Color
    bgcolor: color.Color
    origin: bool
    pad: bool
    ucs4: bool
    resolution: int
    def __init__(
        self,
        file: Union[str, IO[bytes]],
        size: Optional[float] = ...,
        font_index: Optional[int] = ...,
        resolution: Optional[int] = ...,
        ucs4: Optional[int] = ...,  # ucs4 is a bool but passed in as an int
    ) -> None: ...
    def get_rect(
        self, text: str, style: Optional[int] = ..., rotation: Optional[int] = ..., size: Optional[float] = ...,
    ) -> rect.Rect: ...
    def get_metrics(self, text: str, size: Optional[float] = ...) -> List[Tuple[int, int, int, int, float, float]]: ...
    def get_sized_ascender(self, size: float) -> int: ...
    def get_sized_descender(self, size: float) -> int: ...
    def get_sized_height(self, size: float) -> int: ...
    def get_sized_glyph_height(self, size: float) -> int: ...
    def get_sizes(self) -> List[Tuple[int, int, int, float, float]]: ...
    def render(
        self,
        text: str,
        fgcolor: Optional[_ColorValue] = ...,
        bgcolor: Optional[_ColorValue] = ...,
        style: Optional[int] = ...,
        rotation: Optional[int] = ...,
        size: Optional[float] = ...,
    ) -> Tuple[surface.Surface, rect.Rect]: ...
    def render_to(
        self,
        surf: surface.Surface,
        dest: Union[Sequence[int], rect.Rect],
        text: str,
        fgcolor: Optional[_ColorValue] = ...,
        bgcolor: Optional[_ColorValue] = ...,
        style: Optional[int] = ...,
        rotation: Optional[int] = ...,
        size: Optional[float] = ...,
    ) -> rect.Rect: ...
    def render_raw(
        self,
        text: str,
        style: Optional[int] = ...,
        rotation: Optional[int] = ...,
        size: Optional[float] = ...,
        invert: Optional[bool] = ...,
    ) -> Tuple[bytes, Tuple[int, int]]: ...
    def render_raw_to(
        self,
        array: Any,  # BufferProxy
        text: str,
        dest: Optional[Union[Tuple[int, int], Sequence[int]]] = ...,
        style: Optional[int] = ...,
        rotation: Optional[int] = ...,
        size: Optional[float] = ...,
        invert: Optional[bool] = ...,
    ) -> rect.Rect: ...
