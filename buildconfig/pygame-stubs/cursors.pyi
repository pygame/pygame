from typing import Iterator, List, Tuple, Sequence, Optional, Iterable, Union, overload

from pygame.surface import Surface

_Small_string = Tuple[
    str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str
]
_Big_string = Tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
]

arrow: Cursor
diamond: Cursor
broken_x: Cursor
tri_left: Cursor
tri_right: Cursor
thickarrow_strings: _Big_string
sizer_x_strings: _Small_string
sizer_y_strings: _Big_string
sizer_xy_strings: _Small_string

def compile(
    strings: Sequence[str],
    black: str = "X",
    white: str = ".",
    xor: str = "o",
) -> Tuple[Sequence[int], Sequence[int]]: ...
def load_xbm(
    cursorfile: str, maskfile: str
) -> Tuple[List[int], List[int], Tuple[int, ...], Tuple[int, ...]]: ...

class Cursor(Iterable[object]):
    @overload
    def __init__(self, constant: int) -> None: ...
    @overload
    def __init__(
        self,
        size: Union[Tuple[int, int], List[int]],
        hotspot: Union[Tuple[int, int], List[int]],
        xormasks: Sequence[int],
        andmasks: Sequence[int],
    ) -> None: ...
    @overload
    def __init__(
        self,
        hotspot: Union[Tuple[int, int], List[int]],
        surface: Surface,
    ) -> None: ...
    def __iter__(self) -> Iterator[object]: ...
    type: str
    data: Union[
        Tuple[int],
        Tuple[
            Union[Tuple[int, int], List[int]],
            Union[Tuple[int, int], List[int]],
            Sequence[int],
            Sequence[int],
        ],
        Tuple[int, Surface],
    ]
