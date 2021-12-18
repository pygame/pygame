from typing import Any, List, Optional, Sequence, Text, Tuple, Union, overload

from pygame.bufferproxy import BufferProxy
from pygame.math import Vector2
from pygame.rect import Rect

from ._common import CanBeRect, ColorValue, Coordinate, RgbaOutput

class Surface(object):
    _pixels_address: int
    @overload
    def __init__(
        self,
        size: Coordinate,
        flags: int = ...,
        depth: int = ...,
        masks: Optional[ColorValue] = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        size: Coordinate,
        flags: int = ...,
        surface: Surface = ...,
    ) -> None: ...
    def blit(
        self,
        source: Surface,
        dest: Union[Coordinate, CanBeRect],
        area: Optional[CanBeRect] = ...,
        special_flags: int = ...,
    ) -> Rect: ...
    def blits(
        self,
        blit_sequence: Sequence[
            Union[
                Tuple[Surface, Union[Coordinate, CanBeRect]],
                Tuple[Surface, Union[Coordinate, CanBeRect], Union[CanBeRect, int]],
                Tuple[Surface, Union[Coordinate, CanBeRect], CanBeRect, int],
            ]
        ],
        doreturn: Union[int, bool] = 1,
    ) -> Union[List[Rect], None]: ...
    @overload
    def convert(self, surface: Surface) -> Surface: ...
    @overload
    def convert(self, depth: int, flags: int = ...) -> Surface: ...
    @overload
    def convert(self, masks: ColorValue, flags: int = ...) -> Surface: ...
    @overload
    def convert(self) -> Surface: ...
    @overload
    def convert_alpha(self, surface: Surface) -> Surface: ...
    @overload
    def convert_alpha(self) -> Surface: ...
    def copy(self) -> Surface: ...
    def fill(
        self,
        color: ColorValue,
        rect: Optional[CanBeRect] = ...,
        special_flags: int = ...,
    ) -> Rect: ...
    def scroll(self, dx: int = ..., dy: int = ...) -> None: ...
    @overload
    def set_colorkey(self, color: ColorValue, flags: int = ...) -> None: ...
    @overload
    def set_colorkey(self, color: None) -> None: ...
    def get_colorkey(self) -> Optional[RgbaOutput]: ...
    @overload
    def set_alpha(self, value: int, flags: int = ...) -> None: ...
    @overload
    def set_alpha(self, value: None) -> None: ...
    def get_alpha(self) -> Optional[int]: ...
    def lock(self) -> None: ...
    def unlock(self) -> None: ...
    def mustlock(self) -> bool: ...
    def get_locked(self) -> bool: ...
    def get_locks(self) -> Tuple[Any, ...]: ...
    def get_at(self, x_y: Sequence[int]) -> RgbaOutput: ...
    def set_at(self, x_y: Sequence[int], color: ColorValue) -> None: ...
    def get_at_mapped(self, x_y: Sequence[int]) -> int: ...
    def get_palette(self) -> List[RgbaOutput]: ...
    def get_palette_at(self, index: int) -> RgbaOutput: ...
    def set_palette(self, palette: List[ColorValue]) -> None: ...
    def set_palette_at(self, index: int, color: ColorValue) -> None: ...
    def map_rgb(self, color: ColorValue) -> int: ...
    def unmap_rgb(self, mapped_int: int) -> RgbaOutput: ...
    def set_clip(self, rect: Optional[CanBeRect]) -> None: ...
    def get_clip(self) -> Rect: ...
    @overload
    def subsurface(self, rect: Union[CanBeRect, Rect]) -> Surface: ...
    @overload
    def subsurface(
        self,
        left_top: Union[List[float], Tuple[float, float], Vector2],
        width_height: Union[List[float], Tuple[float, float], Vector2],
    ) -> Surface: ...
    @overload
    def subsurface(
        self, left: float, top: float, width: float, height: float
    ) -> Surface: ...
    def get_parent(self) -> Surface: ...
    def get_abs_parent(self) -> Surface: ...
    def get_offset(self) -> Tuple[int, int]: ...
    def get_abs_offset(self) -> Tuple[int, int]: ...
    def get_size(self) -> Tuple[int, int]: ...
    def get_width(self) -> int: ...
    def get_height(self) -> int: ...
    def get_rect(self, **kwargs: Any) -> Rect: ...
    def get_bitsize(self) -> int: ...
    def get_bytesize(self) -> int: ...
    def get_flags(self) -> int: ...
    def get_pitch(self) -> int: ...
    def get_masks(self) -> RgbaOutput: ...
    def set_masks(self, color: ColorValue) -> None: ...
    def get_shifts(self) -> RgbaOutput: ...
    def set_shifts(self, color: ColorValue) -> None: ...
    def get_losses(self) -> RgbaOutput: ...
    def get_bounding_rect(self, min_alpha: int = ...) -> Rect: ...
    def get_view(self, kind: Text = ...) -> BufferProxy: ...
    def get_buffer(self) -> BufferProxy: ...
