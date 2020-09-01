from typing import IO, Literal, Optional, Sequence, Tuple, Union

from . import bufferproxy, surface

_BufferStyle = Union[bufferproxy.BufferProxy, bytes, bytearray, memoryview]
_to_string_format = Literal["p", "RGB", "RGBX", "RGBA", "ARGB", "RGBA_PREMULT", "ARGB_PREMULT"]
_from_buffer_format = Literal["p", "RGB", "BRG", "RGBX", "RGBA", "ARGB"]
_from_string_format = Literal["p", "RGB", "RGBX", "RGBA", "ARGB"]

def load(filename: Union[str, IO[bytes]], namehint: Optional[str] = ...) -> surface.Surface: ...
def save(surface: surface.Surface, filename: Union[str, IO[bytes]]) -> None: ...
def get_extended() -> bool: ...
def tostring(surface: surface.Surface, format: _to_string_format, flipped: Optional[bool] = ...) -> str: ...
def fromstring(
    string: str, size: Union[Sequence[int], Tuple[int, int]], format: _from_string_format, flipped: Optional[bool] = ...,
) -> surface.Surface: ...
def frombuffer(
    bytes: _BufferStyle, size: Union[Sequence[int], Tuple[int, int]], format: _from_buffer_format
) -> surface.Surface: ...

