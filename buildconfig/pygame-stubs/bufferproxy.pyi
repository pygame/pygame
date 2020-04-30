from typing import Any, overload, Optional


class BufferProxy(object):
    parent: Any
    length: int
    raw: str
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, parent: Any) -> None: ...
    def write(self, buffer: bytes, offset: Optional[int]=0) -> None: ...
