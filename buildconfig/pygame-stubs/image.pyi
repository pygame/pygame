from typing import Optional, Tuple, List, Union, IO, Literal

from pygame.surface import Surface
from pygame.bufferproxy import BufferProxy

_BufferStyle = Union[BufferProxy, bytes, bytearray, memoryview]
_to_string_format = Literal['p', 'RGB', 'RGBX', 'RGBA', 'ARGB', 'RGBA_PREMULT', 'ARGB_PREMULT']
_from_buffer_format = Literal['p', 'RGB', 'BRG', 'RGBX', 'RGBA', 'ARGB']
_from_string_format = Literal['p', 'RGB', 'RGBX', 'RGBA', 'ARGB']

def load(filename: Union[str, IO], namehint: Optional[str] = "") -> Surface: ...
def save(surface: Surface, filename: Union[str, IO],
        namehint: Optional[str] = "") -> None: ...
def get_sdl_image_version() -> Union[None, Tuple[int, int, int]]: ...
def get_extended() -> bool: ...
def tostring(surface: Surface, format: _to_string_format,
             flipped: Optional[bool] = False) -> str: ...
def fromstring(
    string: str,
    size: Union[List[int], Tuple[int, int]],
    format: _from_string_format,
    flipped: Optional[bool] = False,
) -> Surface: ...
def frombuffer(
    bytes: _BufferStyle, size: Union[List[int], Tuple[int, int]], format: _from_buffer_format
) -> Surface: ...
