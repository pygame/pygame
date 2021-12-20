from typing import Any, overload

class BufferProxy(object):
    parent: Any
    length: int
    def __init__(self, parent: Any) -> None: ...
    def write(self, buffer: bytes, offset: int = 0) -> None: ...
