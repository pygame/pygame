from typing import Optional, Tuple, List, Union, IO

from pygame.surface import Surface
from pygame.bufferproxy import BufferProxy

_BufferStyle = Union[BufferProxy, bytes, bytearray, memoryview]

def load(filename: Union[str, IO], namehint: Optional[str] = "") -> Surface: ...
def save(surface: Surface, filename: str) -> None: ...
def get_extended() -> bool: ...
def tostring(surface: Surface, format: str, flipped: Optional[bool] = False) -> str: ...
def fromstring(
    string: str,
    size: Union[List[int], Tuple[int, int]],
    format: str,
    flipped: Optional[bool] = False,
) -> Surface: ...
def frombuffer(
    bytes: _BufferStyle, size: Union[List[int], Tuple[int, int]], format: str
) -> Surface: ...
