from typing import Text, Tuple, Union, overload

class Color(object):
    r: int
    g: int
    b: int
    a: int
    cmy: Tuple[float, float, float]
    hsva: Tuple[float, float, float, float]
    hsla: Tuple[float, float, float, float]
    i1i2i3: Tuple[float, float, float]
    @overload
    def __init__(self, name: Text) -> None: ...
    @overload
    def __init__(self, r: int, g: int, b: int, a: int = ...) -> None: ...
    @overload
    def __init__(self, rgbvalue: Union[Text, int]) -> None: ...
    def normalize(self) -> Tuple[float, float, float, float]: ...
    def correct_gamma(self, gamma: float) -> Color: ...
    def set_length(self, len: int) -> None: ...
