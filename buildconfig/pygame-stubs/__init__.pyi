from typing import Any, Tuple, Callable, Union, Optional, overload, Text, Sequence, List

#Most useful stuff
from pygame.constants import *
import pygame.surface
import pygame.rect
import pygame.color
import pygame.event
import pygame.bufferproxy
import pygame.draw
import pygame.display
import pygame.font
import pygame.image
import pygame.key
import pygame.mixer
import pygame.mouse
import pygame.time
import pygame.version

# Advanced stuff
import pygame.cursors
import pygame.joystick
import pygame.mask
import pygame.sprite
import pygame.transform
import pygame.bufferproxy
import pygame.pixelarray
import pygame.pixelcopy
import pygame.sndarray
import pygame.surfarray
import pygame.math
# Other
import pygame.scrap

# This classes are auto imported with pygame, so I put their declaration here
class Rect(pygame.rect.Rect): ...
class Surface(pygame.surface.Surface): ...
class Color(pygame.color.Color): ...
class PixelArray(pygame.pixelarray.PixelArray): ...
class Vector2(pygame.math.Vector2): ...
class Vector3(pygame.math.Vector3): ...


def init() -> Tuple[int, int]: ...
def quit() -> None: ...
def get_init() -> bool: ...
class error(RuntimeError): RuntimeError
def get_error() -> str: ...
def set_error(error_msg: str) -> None: ...
def get_sdl_version() -> Tuple[int, int, int]: ...
def get_sdl_byteorder() -> int: ...
@overload
def encode_string(obj: Union[str, bytes], encoding: Optional[str]='unicode_escape', errors: Optional[str]='backslashreplace', etype: Optional[Exception]=UnicodeEncodeError) -> bytes: ...
@overload
def encode_string(obj: Any, encoding: Optional[str]='unicode_escape', errors: Optional[str]='backslashreplace', etype: Optional[Exception]=UnicodeEncodeError) -> None: ...
@overload
def encode_file_path(obj: Union[str, bytes], etype: Optional[Exception]=UnicodeEncodeError) -> bytes: ...
@overload
def encode_file_path(obj: Any, etype: Optional[Exception]=UnicodeEncodeError) -> bytes: ...
def register_quit(callable: Callable) -> None: ...

# def __getattr__(name) -> Any: ...  # don't error on missing stubs
