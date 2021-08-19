from typing import Any, Tuple, Callable, Union, Optional, overload, Type

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

# These classes are auto imported with pygame, so I put their declaration here
class Rect(rect.Rect): ...
class Surface(surface.Surface): ...
class Color(color.Color): ...
class PixelArray(pixelarray.PixelArray): ...
class Vector2(math.Vector2): ...
class Vector3(math.Vector3): ...
class Cursor(cursors.Cursor): ...

def init() -> Tuple[int, int]: ...
def quit() -> None: ...
def get_init() -> bool: ...

class error(RuntimeError): ...

def get_error() -> str: ...
def set_error(error_msg: str) -> None: ...
def get_sdl_version() -> Tuple[int, int, int]: ...
def get_sdl_byteorder() -> int: ...
def encode_string(
    obj: Union[str, bytes],
    encoding: Optional[str] = "unicode_escape",
    errors: Optional[str] = "backslashreplace",
    etype: Optional[Type[Exception]] = UnicodeEncodeError,
) -> bytes: ...
@overload
def encode_file_path(
    obj: Union[str, bytes], etype: Optional[Type[Exception]] = UnicodeEncodeError
) -> bytes: ...
@overload
def encode_file_path(
    obj: Any, etype: Optional[Type[Exception]] = UnicodeEncodeError
) -> bytes: ...
def register_quit(callable: Callable[[], Any]) -> None: ...
def __getattr__(name: str) -> Any: ...  # don't error on missing stubs
