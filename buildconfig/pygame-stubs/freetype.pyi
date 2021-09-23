import sys
from typing import Tuple, Optional, Union, List, Text, IO, Sequence, Any, Iterable

if sys.version_info >= (3, 6):
    from os import PathLike

    AnyPath = Union[str, bytes, PathLike[str], PathLike[bytes]]
else:
    AnyPath = Union[Text, bytes]

from pygame.surface import Surface
from pygame.color import Color
from pygame.rect import Rect
from pygame.font import Font

_ColorValue = Union[
    Color, str, Tuple[int, int, int], List[int], int, Tuple[int, int, int, int]
]

def get_error() -> str: ...
def get_version() -> Tuple[int, int, int]: ...
def init(cache_size: int = 64, resolution: int = 72) -> None: ...
def quit() -> None: ...
def get_init() -> bool: ...
def was_init() -> bool: ...
def get_cache_size() -> int: ...
def get_default_resolution() -> int: ...
def set_default_resolution(resolution: int) -> None: ...
def SysFont(
    name: Union[str, bytes, Iterable[Union[str, bytes]]],
    size: int,
    bold: int = False,
    italic: int = False,
) -> Font: ...
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
        file: Union[AnyPath, IO[Any], None],
        size: float = 0,
        font_index: int = 0,
        resolution: int = 0,
        ucs4: int = False,
    ) -> None: ...
    def get_rect(
        self,
        text: str,
        style: int = STYLE_DEFAULT,
        rotation: int = 0,
        size: float = 0,
    ) -> Rect: ...
    def get_metrics(
        self, text: str, size: float = 0
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
        style: int = STYLE_DEFAULT,
        rotation: int = 0,
        size: float = 0,
    ) -> Tuple[Surface, Rect]: ...
    def render_to(
        self,
        surf: Surface,
        dest: Union[Tuple[int, int], Sequence[int], Rect],
        text: str,
        fgcolor: Optional[_ColorValue] = None,
        bgcolor: Optional[_ColorValue] = None,
        style: int = STYLE_DEFAULT,
        rotation: int = 0,
        size: float = 0,
    ) -> Rect: ...
    def render_raw(
        self,
        text: str,
        style: int = STYLE_DEFAULT,
        rotation: int = 0,
        size: float = 0,
        invert: bool = False,
    ) -> Tuple[bytes, Tuple[int, int]]: ...
    def render_raw_to(
        self,
        array: Any,
        text: str,
        dest: Optional[Union[Tuple[int, int], List[int]]] = None,
        style: int = STYLE_DEFAULT,
        rotation: int = 0,
        size: float = 0,
        invert: bool = False,
    ) -> Rect: ...
