from typing import Any, Tuple, Callable, Optional, overload, Type

# Re-export modules as members; see PEP 484 for stub export rules.

# Most useful stuff
from pygame.constants import *
from pygame import surface as surface
from pygame import rect as rect
from pygame import color as color
from pygame import event as event
from pygame import draw as draw
from pygame import display as display
from pygame import font as font
from pygame import image as image
from pygame import key as key
from pygame import mixer as mixer
from pygame import mouse as mouse
from pygame import time as time
from pygame import version as version

# Advanced stuff
from pygame import cursors as cursors
from pygame import joystick as joystick
from pygame import mask as mask
from pygame import sprite as sprite
from pygame import transform as transform
from pygame import bufferproxy as bufferproxy
from pygame import pixelarray as pixelarray
from pygame import pixelcopy as pixelcopy
from pygame import sndarray as sndarray
from pygame import surfarray as surfarray
from pygame import math as math
from pygame import fastevent as fastevent

# Other
from pygame import scrap as scrap

from ._common import _AnyPath

# These classes are auto imported with pygame, so I put their declaration here
from pygame.rect import Rect
from pygame.surface import Surface
from pygame.color import Color
from pygame.pixelarray import PixelArray
from pygame.math import Vector2, Vector3
from pygame.cursors import Cursor
from pygame.bufferproxy import BufferProxy
from pygame.mask import Mask

__all__ = [
    "Rect",
    "Surface",
    "Color",
    "PixelArray",
    "Vector2",
    "Vector3",
    "Cursor",
    "BufferProxy",
    "Mask",
]

def init() -> Tuple[int, int]: ...
def quit() -> None: ...
def get_init() -> bool: ...

class error(RuntimeError): ...

def get_error() -> str: ...
def set_error(error_msg: str) -> None: ...
def get_sdl_version() -> Tuple[int, int, int]: ...
def get_sdl_byteorder() -> int: ...
def encode_string(
    obj: Optional[_AnyPath],
    encoding: Optional[str] = "unicode_escape",
    errors: Optional[str] = "backslashreplace",
    etype: Optional[Type[Exception]] = UnicodeEncodeError,
) -> bytes: ...
@overload
def encode_file_path(
    obj: Optional[_AnyPath], etype: Optional[Type[Exception]] = UnicodeEncodeError
) -> bytes: ...
@overload
def encode_file_path(
    obj: Any, etype: Optional[Type[Exception]] = UnicodeEncodeError
) -> bytes: ...
def register_quit(callable: Callable[[], Any]) -> None: ...
def __getattr__(name: str) -> Any: ...  # don't error on missing stubs
