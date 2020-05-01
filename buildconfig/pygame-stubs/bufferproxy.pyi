from typing import Any, overload, Optional, TypeVar, Text

AnyStr = TypeVar("AnyStr", Text, bytes)

class BufferProxy(object):
    parent: Any
    length: int
    raw: AnyStr
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, parent: Any) -> None: ...
    def write(self, buffer: bytes, offset: Optional[int] = 0) -> None: ...
