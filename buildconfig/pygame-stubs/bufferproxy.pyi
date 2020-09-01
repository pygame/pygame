from typing import AnyStr, Optional

class BufferProxy(object):
    parent: Optional[BufferProxy]
    length: int
    raw: AnyStr
    def __init__(self, parent: Optional[BufferProxy] = ...) -> None: ...
    def write(self, buffer: bytes, offset: Optional[int] = ...) -> None: ...

