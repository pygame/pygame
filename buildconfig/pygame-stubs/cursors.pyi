from typing import Tuple, Sequence, Optional

_Bitmap = Tuple[
    Tuple[int, int],
    Tuple[int, int],
    Tuple[
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
    ],
    Tuple[
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
    ],
]
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

arrow: _Bitmap
diamond: _Bitmap
broken_x: _Bitmap
tri_left: _Bitmap
tri_right: _Bitmap
thickarrow_strings: _Big_string
sizer_x_strings: _Small_string
sizer_y_strings: _Big_string
sizer_xy_strings: _Small_string

def compile(
    strings: Sequence[str],
    black: Optional[str] = "X",
    white: Optional[str] = ".",
    xor="o",
) -> Tuple[Sequence[int], Sequence[int]]: ...
def load_xbm(cursorfile: str, maskfile: str): ...
