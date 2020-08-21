from typing import IO, Any, Optional, Tuple, Union, overload

from numpy import ndarray
from pygame.event import Event

from . import music as music

def init(
    frequency: Optional[int] = ...,
    size: Optional[int] = ...,
    channels: Optional[int] = ...,
    buffer: Optional[int] = ...,
    devicename: Optional[Union[str, None]] = ...,
    allowedchanges: Optional[int] = ...,
) -> None: ...
def pre_init(
    frequency: Optional[int] = ...,
    size: Optional[int] = ...,
    channels: Optional[int] = ...,
    buffer: Optional[int] = ...,
    devicename: Optional[Union[str, None]] = ...,
) -> None: ...
def quit() -> None: ...
def get_init() -> Tuple[int, int, int]: ...
def stop() -> None: ...
def pause() -> None: ...
def unpause() -> None: ...
def fadeout(time: int) -> None: ...
def set_num_channels(count: int) -> None: ...
def get_num_channels() -> int: ...
def set_reserved() -> None: ...
def find_channel(force: bool) -> Channel: ...
def get_busy() -> bool: ...
def get_sdl_mixer_version(linked: bool) -> Tuple[int, int, int]: ...

class Sound:
    @overload
    def __init__(self, file: IO[bytes]) -> None: ...
    @overload
    def __init__(self, buffer: Any) -> None: ...  # Buffer protocol is still not implemented in typing
    @overload
    def __init__(self, array: numpy.ndarray) -> None: ...  # Buffer protocol is still not implemented in typing
    def play(self, loops: Optional[int] = ..., maxtime: Optional[int] = ..., fade_ms: Optional[int] = ...,) -> Channel: ...
    def stop(self) -> None: ...
    def fadeout(self, time: int) -> None: ...
    def set_volume(self, value: float) -> None: ...
    def get_volume(self) -> float: ...
    def get_num_channels(self) -> int: ...
    def get_length(self) -> float: ...
    def get_raw(self) -> bytes: ...

class Channel:
    def __init__(self, id: int) -> None: ...
    def play(
        self, sound: Sound, loops: Optional[int] = ..., maxtime: Optional[int] = ..., fade_ms: Optional[int] = ...,
    ) -> None: ...
    def stop(self) -> None: ...
    def pause(self) -> None: ...
    def unpause(self) -> None: ...
    def fadeout(self, time: int) -> None: ...
    @overload
    def set_volume(self, value: float) -> None: ...
    @overload
    def set_volume(self, left: float, right: float) -> None: ...
    def get_volume(self) -> float: ...
    def get_busy(self) -> bool: ...
    def get_sound(self) -> Sound: ...
    def get_queue(self) -> Sound: ...
    def set_endevent(self, type: Optional[Union[int, Event]] = ...) -> None: ...
    def get_endevent(self) -> int: ...

