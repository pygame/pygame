import sys
from typing import Any, Optional, Tuple, List, Union, IO, Literal

from pygame.surface import Surface
from pygame.bufferproxy import BufferProxy

if sys.version_info >= (3, 6):
    from os import PathLike

    AnyPath = Union[str, bytes, PathLike[str], PathLike[bytes]]
else:
    AnyPath = Union[Text, bytes]

_BufferStyle = Union[BufferProxy, bytes, bytearray, memoryview]
_to_string_format = Literal[
    "p", "RGB", "RGBX", "RGBA", "ARGB", "RGBA_PREMULT", "ARGB_PREMULT"
]
_from_buffer_format = Literal["p", "RGB", "BGR", "RGBX", "RGBA", "ARGB"]
_from_string_format = Literal["p", "RGB", "RGBX", "RGBA", "ARGB"]

def load(filename: Union[AnyPath, IO[Any]], namehint: str = "") -> Surface: ...
def save(
    surface: Surface, filename: Union[AnyPath, IO[Any]], namehint: str = ""
) -> None: ...
def get_sdl_image_version() -> Union[None, Tuple[int, int, int]]: ...
def get_extended() -> bool: ...
def tostring(
    surface: Surface, format: _to_string_format, flipped: bool = False
) -> str: ...
def fromstring(
    string: str,
    size: Union[List[int], Tuple[int, int]],
    format: _from_string_format,
    flipped: bool = False,
) -> Surface: ...
def frombuffer(
    bytes: _BufferStyle,
    size: Union[List[int], Tuple[int, int]],
    format: _from_buffer_format,
) -> Surface: ...
