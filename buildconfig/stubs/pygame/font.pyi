from typing import Callable, Hashable, Iterable, List, Optional, Tuple, Union

from pygame.surface import Surface

from ._common import ColorValue, FileArg, Literal

# TODO: Figure out a way to type this attribute such that mypy knows it's not
# always defined at runtime
UCS4: Literal[1]

def init() -> None: ...
def quit() -> None: ...
def get_init() -> bool: ...
def get_ttf_font() -> Tuple[int, int, int]: ...
def get_default_font() -> str: ...
def get_fonts() -> List[str]: ...
def match_font(
    name: Union[str, bytes, Iterable[Union[str, bytes]]],
    bold: Hashable = False,
    italic: Hashable = False,
) -> str: ...
def SysFont(
    name: Union[str, bytes, Iterable[Union[str, bytes]]],
    size: int,
    bold: Hashable = False,
    italic: Hashable = False,
    constructor: Optional[Callable[[Optional[str], int, bool, bool], Font]] = None,
) -> Font: ...

class Font:
    bold: bool
    italic: bool
    underline: bool
    def __init__(self, name: Optional[FileArg], size: int) -> None: ...
    def render(
        self,
        text: Union[str, bytes, None],
        antialias: bool,
        color: ColorValue,
        background: Optional[ColorValue] = None,
    ) -> Surface: ...
    def size(self, text: Union[str, bytes]) -> Tuple[int, int]: ...
    def set_underline(self, value: bool) -> None: ...
    def get_underline(self) -> bool: ...
    def set_bold(self, value: bool) -> None: ...
    def get_bold(self) -> bool: ...
    def set_italic(self, value: bool) -> None: ...
    def metrics(
        self, text: Union[str, bytes]
    ) -> List[Tuple[int, int, int, int, int]]: ...
    def get_italic(self) -> bool: ...
    def get_linesize(self) -> int: ...
    def get_height(self) -> int: ...
    def get_ascent(self) -> int: ...
    def get_descent(self) -> int: ...

FontType = Font
