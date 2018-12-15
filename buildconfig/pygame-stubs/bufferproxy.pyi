from typing import Any, overload

class BufferProxy(object):
    parent: Any
    length: int
    raw: bytes
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, parent: Any) -> None: ...
    def write(self, buffer: bytes, offset: int = ...) -> None: ...
